using Flux
using Random
using Plots
using Statistics

# ===========================
# Parâmetros do modelo SIR
# ===========================
β = 0.3f0
γ = 0.1f0

S0 = 0.9f0
I0 = 0.1f0
R0 = 0.0f0

# ===========================
# Rede neural
# ===========================
Random.seed!(1234)

pinn = Chain(
    Dense(1, 16, tanh),
    Dense(16, 16, tanh),
    Dense(16, 3)  # saída: S, I, R
)

# ===========================
# Função para diferenças finitas
# ===========================
function compute_residuals(model, t::Float32)
    h = 0.001f0
    
    # Calcular em três pontos: t-h, t, t+h
    y_minus = model([t - h])
    y = model([t])
    y_plus = model([t + h])
    
    # Extrair valores
    S_minus, I_minus, R_minus = y_minus[1], y_minus[2], y_minus[3]
    S, I, R = y[1], y[2], y[3]
    S_plus, I_plus, R_plus = y_plus[1], y_plus[2], y_plus[3]
    
    # Derivadas por diferenças centrais
    dS_dt = (S_plus - S_minus) / (2h)
    dI_dt = (I_plus - I_minus) / (2h)
    dR_dt = (R_plus - R_minus) / (2h)
    
    # Resíduos das equações SIR
    res1 = dS_dt + β * S * I
    res2 = dI_dt - β * S * I + γ * I
    res3 = dR_dt - γ * I
    
    # Erro de normalização
    norm_error = S + I + R - 1.0f0
    
    return res1^2 + res2^2 + res3^2 + 10.0f0 * norm_error^2
end

# ===========================
# Função de perda completa
# ===========================
function total_loss(model, t_points)
    loss = 0.0f0
    
    # 1. Condição inicial
    y0 = model([0.0f0])
    loss += 1000.0f0 * (
        (y0[1] - S0)^2 +
        (y0[2] - I0)^2 +
        (y0[3] - R0)^2
    )
    
    # 2. Perda da física nos pontos internos
    for t in t_points
        t_f32 = Float32(t)
        loss += compute_residuals(model, t_f32)
    end
    
    return loss / (length(t_points) + 1)
end

# ===========================
# Treinamento
# ===========================
println("Iniciando treinamento da PINN SIR...")

# Pontos de treinamento
t_train = collect(0:0.1:10.0)

# Configuração do otimizador
opt = Flux.setup(Flux.Adam(0.01), pinn)

# Histórico de perda
loss_history = Float32[]

# Loop de treinamento
for epoch in 1:2000
    # Calcular gradiente
    loss_val, grads = Flux.withgradient(pinn) do model
        total_loss(model, t_train)
    end
    
    # Atualizar parâmetros
    Flux.update!(opt, pinn, grads[1])
    
    # Mostrar progresso
    if epoch % 100 == 0
        push!(loss_history, loss_val)
        println("Epoch $epoch, Loss = $loss_val")
    end
end

# ===========================
# Gerar predições para gráficos
# ===========================
println("\nGerando dados para gráficos...")

t_plot = 0:0.05:10
S_pred = Float32[]
I_pred = Float32[]
R_pred = Float32[]
total_pop = Float32[]

# Também calcular derivadas para análise
dS_dt_vals = Float32[]
dI_dt_vals = Float32[]
dR_dt_vals = Float32[]
residuals = Float32[]  # Resíduos das equações

h = 0.001f0

for t in t_plot
    t_f32 = Float32(t)
    
    # Valores principais
    y = pinn([t_f32])
    push!(S_pred, y[1])
    push!(I_pred, y[2])
    push!(R_pred, y[3])
    push!(total_pop, y[1] + y[2] + y[3])
    
    # Calcular derivadas
    y_minus = pinn([t_f32 - h])
    y_plus = pinn([t_f32 + h])
    
    dS_dt = (y_plus[1] - y_minus[1]) / (2h)
    dI_dt = (y_plus[2] - y_minus[2]) / (2h)
    dR_dt = (y_plus[3] - y_minus[3]) / (2h)
    
    push!(dS_dt_vals, dS_dt)
    push!(dI_dt_vals, dI_dt)
    push!(dR_dt_vals, dR_dt)
    
    # Calcular resíduo (erro nas equações)
    res = abs(dS_dt + β * y[1] * y[2]) +
           abs(dI_dt - β * y[1] * y[2] + γ * y[2]) +
           abs(dR_dt - γ * y[2])
    push!(residuals, res)
end

# ===========================
# CRIAR OS GRÁFICOS
# ===========================
println("Criando gráficos...")

# GRÁFICO 1: Evolução temporal da população
p1 = plot(t_plot, S_pred, 
          label="Suscetíveis (S)", 
          linewidth=3, 
          color=:blue,
          legend=:right)

plot!(p1, t_plot, I_pred, 
      label="Infectados (I)", 
      linewidth=3, 
      color=:red)

plot!(p1, t_plot, R_pred, 
      label="Recuperados (R)", 
      linewidth=3, 
      color=:green)

# Linha para mostrar população total (deve ser ≈1)
plot!(p1, t_plot, total_pop, 
      label="Total S+I+R", 
      linewidth=2, 
      linestyle=:dash,
      color=:black,
      alpha=0.7)

xlabel!(p1, "Tempo (dias)")
ylabel!(p1, "Proporção da população")
title!(p1, "Evolução da Epidemia - Modelo SIR")

# GRÁFICO 2: Derivadas (velocidades de mudança)
p2 = plot(t_plot, dS_dt_vals, 
          label="dS/dt", 
          linewidth=2, 
          color=:blue,
          legend=:topright)

plot!(p2, t_plot, dI_dt_vals, 
      label="dI/dt", 
      linewidth=2, 
      color=:red)

plot!(p2, t_plot, dR_dt_vals, 
      label="dR/dt", 
      linewidth=2, 
      color=:green)

xlabel!(p2, "Tempo (dias)")
ylabel!(p2, "Taxa de variação")
title!(p2, "Derivadas - Como cada grupo muda")

# GRÁFICO 3: Resíduos das equações e perda
p3 = plot(t_plot, residuals,
          label="Erro nas equações",
          linewidth=2,
          color=:purple,
          fillrange=0,
          fillalpha=0.2,
          legend=:topright)

xlabel!(p3, "Tempo (dias)")
ylabel!(p3, "Erro (resíduo)")
title!(p3, "Precisão da PINN nos pontos")

# GRÁFICO 4: Histórico de treinamento
p4 = plot(100:100:2000, loss_history,
          label="Perda",
          linewidth=2,
          color=:orange,
          marker=:circle,
          markersize=3,
          xlabel="Época",
          ylabel="Perda",
          title="Histórico de Treinamento",
          yscale=:log10)

# ===========================
# MOSTRAR TODOS OS GRÁFICOS
# ===========================
# Layout: 2x2 (quatro gráficos)
plot(p1, p2, p3, p4, 
     layout=(2, 2), 
     size=(1200, 800),
     plot_title="PINN - Modelo SIR: Resultados Completos")

# Salvar o gráfico (opcional)
savefig("sir_pinn_resultados.png")
println("\nGráfico salvo como 'sir_pinn_resultados.png'")

# ===========================
# Análise dos resultados
# ===========================
println("\n" * "="^50)
println("ANÁLISE DOS RESULTADOS")
println("="^50)

# Encontrar pico da infecção
max_I, idx = findmax(I_pred)
t_max = t_plot[idx]
println("\n📊 Pico da epidemia:")
println("   Tempo do pico: $t_max dias")
println("   Máximo de infectados: $(round(max_I*100, digits=2))% da população")

# Calcular R0 básico
R0_basic = β / γ
println("\n📈 Parâmetros epidemiológicos:")
println("   R₀ básico = β/γ = $β/$γ = $(round(R0_basic, digits=2))")

# Verificar conservação da população
avg_total = mean(total_pop)
max_deviation = maximum(abs.(total_pop .- 1.0f0))
println("\n✅ Verificação de consistência:")
println("   População média (S+I+R): $(round(avg_total, digits=4))")
println("   Desvio máximo de 1: $(round(max_deviation, digits=6))")

# Calcular algumas métricas adicionais
final_S = S_pred[end]
final_R = R_pred[end]
herd_immunity_threshold = 1 - 1/R0_basic

println("\n📊 Estatísticas finais (t=10 dias):")
println("   Suscetíveis finais: $(round(final_S*100, digits=2))%")
println("   Recuperados finais: $(round(final_R*100, digits=2))%")
println("   Limiar de imunidade de rebanho: $(round(herd_immunity_threshold*100, digits=1))%")

# Análise da qualidade do treinamento
initial_loss = loss_history[1]
final_loss = loss_history[end]
reduction = initial_loss / final_loss

println("\n" * "="^50)
println("DESEMPENHO DO TREINAMENTO:")
println("="^50)
println("   Perda inicial: $(round(initial_loss, digits=6))")
println("   Perda final: $(round(final_loss, digits=6))")
println("   Redução: $(round(reduction, digits=1))x")

if final_loss < 0.001
    println("   🎉 EXCELENTE! A PINN aprendeu muito bem o modelo SIR.")
elseif final_loss < 0.01
    println("   ✅ BOM! A PINN aprendeu bem o modelo SIR.")
else
    println("   ⚠️  RAZOÁVEL. A PINN capturou a tendência geral.")
end

# Explicação dos resultados
println("\n" * "="^50)
println("INTERPRETAÇÃO DOS RESULTADOS:")
println("="^50)
println("1. 🟦 S (Suscetíveis): Diminui de 90% para $(round(final_S*100, digits=1))%")
println("2. 🟥 I (Infectados): Pico de $(round(max_I*100, digits=1))% no dia $t_max")
println("3. 🟩 R (Recuperados): Aumenta de 0% para $(round(final_R*100, digits=1))%")
println("4. 📊 R₀ = 3.0: Cada pessoa infecta 3 outras em média")
println("5. 🛡️  Imunidade de rebanho: $(round(herd_immunity_threshold*100, digits=1))% da população")

println("\n💡 OBSERVAÇÕES:")
println("   • Com R₀ = 3.0, a epidemia se espalha rapidamente")
println("   • O pico ocorre quando dI/dt = 0 (taxa máxima de infecção)")
println("   • A PINN resolveu as equações diferenciais com erro de $(round(mean(residuals), digits=6))")

println("\n✅ Treinamento concluído com sucesso! ✅")
println("📁 Gráfico salvo como: sir_pinn_resultados.png")
