using ForwardDiff
using Random
using Plots
using Plots.PlotMeasures
using Statistics
using Printf
using JLD2
using Optim

# ===========================
# 1. PARÂMETROS
# ===========================
RETREINAR = false   # ← mude para false após o primeiro treino

# -----------------------------------------------------------------------
# DADOS REAIS — Todas as semanas epidemiológicas de 2024, Botucatu (Dengue)
# Fonte: TABNET / SINAN
# Total anual: 16.404 casos
# Cada valor = número absoluto de casos por semana epidemiológica (INCIDÊNCIA)
# -----------------------------------------------------------------------
I_real_abs = Float64[
    116, 128, 164, 279, 179, 212, 446, 493, 542, 770,
    971, 1276, 1113, 1109, 1358, 1154, 1002, 1018, 1137, 1120,
    713, 372, 161, 152, 133, 68, 34, 21, 7, 15,
    8, 12, 7, 4, 5, 7, 5, 6, 6, 9,
    4, 5, 10, 3, 8, 3, 7, 6, 7, 6, 7, 6
]

C_real_abs = cumsum(I_real_abs)
C_ESCALA   = maximum(C_real_abs)
C_real     = C_real_abs ./ C_ESCALA

I_ESCALA = maximum(I_real_abs)

t_real  = collect(1.0:1.0:52.0)
T_FINAL = 52.0

POP_BOTUCATU = 145_000.0

PERIODO_INFECCIOSO = 1.0
I0      = (I_real_abs[1] * PERIODO_INFECCIOSO) / C_ESCALA
C0      = 0.0
R0_init = 0.0
S0      = 1.0 - I0 - R0_init

ARQUIVO_MODELO = "sir_pinn_dengue.jld2"
ARQUIVO_CKPT   = "sir_pinn_dengue_ckpt.jld2"

# Pesos da loss
W_IC       = 100.0
W_PHYS_F1  = 50.0
W_DADOS_F1 = 50.0

t_collocation = collect(0.0:1.0:T_FINAL)

EPOCHS_ADAM_F1 = 10000
EPOCHS_LBFGS   = 5000

# ===========================
# 2. ARQUITETURA
#    [1 → 16 → 32 → 16 → 4]  (saída: S, I, R, C)
#    β = exp(log_β), γ = exp(log_γ) — parametrização logarítmica
#    N_PARAMS = 1174  (1172 pesos neurais + log_β + log_γ)
# ===========================
const N_PARAMS  = 1174
const N_REDE    = 1172
const IDX_log_β = 1173
const IDX_log_γ = 1174

function inicializar_params()
    Random.seed!(9999)
    params = zeros(Float64, N_PARAMS)
    offset = 0

    lim = sqrt(6.0 / (1 + 16))
    n = 16*1;  params[offset+1:offset+n] = (rand(n).*2lim).-lim;  offset += n
    n = 16;    params[offset+1:offset+n] .= 0.0;                   offset += n

    lim = sqrt(6.0 / (16 + 32))
    n = 32*16; params[offset+1:offset+n] = (rand(n).*2lim).-lim;  offset += n
    n = 32;    params[offset+1:offset+n] .= 0.0;                   offset += n

    lim = sqrt(6.0 / (32 + 16))
    n = 16*32; params[offset+1:offset+n] = (rand(n).*2lim).-lim;  offset += n
    n = 16;    params[offset+1:offset+n] .= 0.0;                   offset += n

    lim = sqrt(6.0 / (16 + 4))
    n = 4*16;  params[offset+1:offset+n] = (rand(n).*2lim).-lim;  offset += n
    n = 4;     params[offset+1:offset+n] .= 0.0;                   offset += n

    params[IDX_log_β] = log(1.5)
    params[IDX_log_γ] = log(0.2)

    return params
end

println("SIR-PINN — Problema Inverso com Dados Reais de Dengue")
println("  Cidade: Botucatu/SP | Ano completo 2024 (52 semanas)")
println("  Arquitetura: [1 → 16 → 32 → 16 → 4]  (saída: S, I, R, C)")
println("  Estágio 1: Adam $(EPOCHS_ADAM_F1) épocas | IC=$(W_IC) Física=$(W_PHYS_F1) Dados=$(W_DADOS_F1)")
println("  Estágio 2: L-BFGS $(EPOCHS_LBFGS) iter  | β e γ CONGELADOS nos valores do Adam")
println("  N_PARAMS = $N_PARAMS  (rede: $N_REDE + β + γ)")
println("="^60)
@printf("  S0=%.6f | I0=%.6f | C0=%.1f\n", S0, I0, C0)
println("="^60)

# ===========================
# 3. FORWARD PASS
# ===========================
σ(x) = 1.0 / (1.0 + exp(-x))

function predict(t::T, params) where T
    offset = 0

    W1 = reshape(params[offset+1:offset+16],  16,  1);  offset += 16
    b1 =         params[offset+1:offset+16];             offset += 16

    W2 = reshape(params[offset+1:offset+512], 32, 16);  offset += 512
    b2 =         params[offset+1:offset+32];             offset += 32

    W3 = reshape(params[offset+1:offset+512], 16, 32);  offset += 512
    b3 =         params[offset+1:offset+16];             offset += 16

    W4 = reshape(params[offset+1:offset+64],   4, 16);  offset += 64
    b4 =         params[offset+1:offset+4]

    t_n = [t / T_FINAL]

    h1 = tanh.(W1 * t_n .+ b1)
    h2 = tanh.(W2 * h1  .+ b2)
    h3 = tanh.(W3 * h2  .+ b3)
    return σ.(W4 * h3  .+ b4)
end

# ===========================
# 4. RK4 — visualização final
# ===========================
function resolver_rk4(β_val, γ_val)
    dt = 0.1; steps = Int(T_FINAL / dt)
    ts = Float64[]
    Ss, Is, Rs, Cs = Float64[], Float64[], Float64[], Float64[]
    u = [S0, I0, R0_init, C0]

    f(u) = [
        -β_val * u[1] * u[2],
         β_val * u[1] * u[2] - γ_val * u[2],
         γ_val * u[2],
         β_val * u[1] * u[2]
    ]

    for i in 0:steps
        push!(ts, i*dt)
        push!(Ss, u[1]); push!(Is, u[2])
        push!(Rs, u[3]); push!(Cs, u[4])
        k1=f(u); k2=f(u.+0.5dt.*k1); k3=f(u.+0.5dt.*k2); k4=f(u.+dt.*k3)
        u = u .+ (dt/6).*(k1.+2k2.+2k3.+k4)
    end
    return ts, Ss, Is, Rs, Cs
end

# ===========================
# 5. RESÍDUOS FÍSICOS — versão Adam (lê β e γ do vetor)
# ===========================
function compute_residuals(t, params)
    β_val = exp(params[IDX_log_β])
    γ_val = exp(params[IDX_log_γ])

    y = predict(t, params)
    S, I, R, C = y[1], y[2], y[3], y[4]

    derivs = ForwardDiff.derivative(τ -> predict(τ, params), t)
    dS, dI, dR, dC = derivs[1], derivs[2], derivs[3], derivs[4]

    res_S = dS - (-β_val * S * I)
    res_I = dI - ( β_val * S * I - γ_val * I)
    res_R = dR - ( γ_val * I)
    res_C = dC - ( β_val * S * I)

    return res_S^2 + res_I^2 + res_R^2 + res_C^2
end

# ===========================
# 5b. RESÍDUOS FÍSICOS — versão L-BFGS (β e γ fixos externos)
# ===========================
function compute_residuals_fixo(t, params_rede, β_fixo, γ_fixo)
    y = predict(t, params_rede)
    S, I, R, C = y[1], y[2], y[3], y[4]

    derivs = ForwardDiff.derivative(τ -> predict(τ, params_rede), t)
    dS, dI, dR, dC = derivs[1], derivs[2], derivs[3], derivs[4]

    res_S = dS - (-β_fixo * S * I)
    res_I = dI - ( β_fixo * S * I - γ_fixo * I)
    res_R = dR - ( γ_fixo * I)
    res_C = dC - ( β_fixo * S * I)

    return res_S^2 + res_I^2 + res_R^2 + res_C^2
end

# ===========================
# 6. FUNÇÕES DE PERDA
# ===========================
function loss_function(params, C_obs, t_obs, w_phys, w_dados)
    y0 = predict(0.0, params)

    loss_ic = W_IC * (
        (y0[1] - S0)^2      +
        (y0[2] - I0)^2      +
        (y0[3] - R0_init)^2 +
        (y0[4] - C0)^2
    )

    loss_phys  = w_phys  * sum(compute_residuals(t, params) for t in t_collocation)
    loss_dados = w_dados * sum(
        (predict(t_obs[i], params)[4] - C_obs[i])^2
        for i in eachindex(t_obs)
    )

    return loss_ic + loss_phys + loss_dados
end

function loss_lbfgs(params_rede, C_obs, t_obs, β_fixo, γ_fixo)
    y0 = predict(0.0, params_rede)

    loss_ic = W_IC * (
        (y0[1] - S0)^2      +
        (y0[2] - I0)^2      +
        (y0[3] - R0_init)^2 +
        (y0[4] - C0)^2
    )

    loss_phys  = W_PHYS_F1 * sum(
        compute_residuals_fixo(t, params_rede, β_fixo, γ_fixo)
        for t in t_collocation
    )
    loss_dados = W_DADOS_F1 * sum(
        (predict(t_obs[i], params_rede)[4] - C_obs[i])^2
        for i in eachindex(t_obs)
    )

    return loss_ic + loss_phys + loss_dados
end

# ===========================
# 7. CARREGAR OU TREINAR
# ===========================
function tentar_carregar_compativel(arquivo)
    dados = JLD2.load(arquivo)
    p_salvo = dados["params"]
    n_salvo = length(p_salvo)

    if n_salvo == N_PARAMS
        println("  Arquitetura compatível ($N_PARAMS params). Carregamento direto.")
        return p_salvo, dados["loss_history"], dados["epochs"]
    else
        println("  Arquitetura desconhecida ($n_salvo params). Iniciando do zero.")
        return nothing, Float64[], 0
    end
end

function executar_treino()
    params       = inicializar_params()
    loss_history = Float64[]

    lr              = 0.001
    m_adam          = zeros(N_PARAMS)
    v_adam          = zeros(N_PARAMS)
    beta1, beta2, ε = 0.9, 0.999, 1e-8

    # -------------------------------------------------------
    # ESTÁGIO 1 — Adam F1
    # -------------------------------------------------------
    println("\n── ESTÁGIO 1: Adam F1 ($(EPOCHS_ADAM_F1) épocas) ──")
    println("  Física=$(W_PHYS_F1) | Dados=$(W_DADOS_F1)")
    println("="^60)

    for epoch in 1:EPOCHS_ADAM_F1
        grad = ForwardDiff.gradient(
            p -> loss_function(p, C_real, t_real, W_PHYS_F1, W_DADOS_F1),
            params
        )

        m_adam .= beta1 .* m_adam .+ (1-beta1) .* grad
        v_adam .= beta2 .* v_adam .+ (1-beta2) .* (grad.^2)
        m̂  = m_adam ./ (1 - beta1^epoch)
        v̂  = v_adam ./ (1 - beta2^epoch)
        params .-= lr .* m̂ ./ (sqrt.(v̂) .+ ε)

        if epoch % 500 == 0
            l     = loss_function(params, C_real, t_real, W_PHYS_F1, W_DADOS_F1)
            β_est = exp(params[IDX_log_β])
            γ_est = exp(params[IDX_log_γ])
            push!(loss_history, l)
            @printf("[Adam F1] Epoch %5d | Loss: %.6f | β=%.4f | γ=%.4f | R0=%.3f\n",
                    epoch, l, β_est, γ_est, β_est/γ_est)

            JLD2.jldsave(ARQUIVO_CKPT;
                params=params, loss_history=loss_history, epochs=epoch)
        end
    end

    β_fixo = exp(params[IDX_log_β])
    γ_fixo = exp(params[IDX_log_γ])
    println("\n✔ Adam F1 concluída.")
    @printf("  β=%.4f | γ=%.4f | R0=%.3f\n", β_fixo, γ_fixo, β_fixo/γ_fixo)
    @printf("  Período infeccioso: %.2f semanas\n", 1.0/γ_fixo)

    # -------------------------------------------------------
    # ESTÁGIO 2 — L-BFGS com β e γ CONGELADOS
    # Refina apenas os 1172 pesos neurais
    # -------------------------------------------------------
    println("\n── ESTÁGIO 2: L-BFGS ($(EPOCHS_LBFGS) iter) — β e γ CONGELADOS ──")
    @printf("  β fixo: %.4f | γ fixo: %.4f | R0: %.3f\n",
            β_fixo, γ_fixo, β_fixo/γ_fixo)
    println("  Otimiza: apenas pesos neurais ($N_REDE params)")
    println("="^60)

    params_rede = copy(params[1:N_REDE])

    f_lbfgs(p) = loss_lbfgs(p, C_real, t_real, β_fixo, γ_fixo)

    function g_lbfgs!(G, p)
        G .= ForwardDiff.gradient(f_lbfgs, p)
        return G
    end

    function cb_lbfgs(state)
        if state.iteration > 0 && state.iteration % 500 == 0
            l = state.value
            push!(loss_history, l)
            @printf("[L-BFGS] Iter  %5d | Loss: %.6f | β=%.4f (fixo) | γ=%.4f (fixo) | R0=%.3f\n",
                    state.iteration, l, β_fixo, γ_fixo, β_fixo/γ_fixo)
        end
        return false
    end

    resultado = Optim.optimize(
        f_lbfgs,
        g_lbfgs!,
        params_rede,
        Optim.LBFGS(),
        Optim.Options(
            iterations = EPOCHS_LBFGS,
            show_trace = false,
            callback   = cb_lbfgs,
            g_tol      = 1e-6
        )
    )

    params_rede_otimizada = Optim.minimizer(resultado)
    params[1:N_REDE]  = params_rede_otimizada
    params[IDX_log_β] = log(β_fixo)
    params[IDX_log_γ] = log(γ_fixo)

    epochs_total = EPOCHS_ADAM_F1 + EPOCHS_LBFGS

    println("\n✔ L-BFGS concluído.")
    println("  Convergiu: $(Optim.converged(resultado))")
    @printf("  β: %.4f (inalterado) | γ: %.4f (inalterado) | R0: %.3f\n",
            β_fixo, γ_fixo, β_fixo/γ_fixo)

    JLD2.jldsave(ARQUIVO_MODELO;
        params=params, loss_history=loss_history, epochs=epochs_total)
    println("\n✔ Modelo salvo em: $ARQUIVO_MODELO")

    return params, loss_history
end

# ===========================
# PONTO DE ENTRADA
# ===========================
if isfile(ARQUIVO_MODELO) && !RETREINAR
    println("\nModelo encontrado. Tentando carregar...")
    result = tentar_carregar_compativel(ARQUIVO_MODELO)
    params, loss_history, epochs_total = result
    if params === nothing
        params, loss_history = executar_treino()
    else
        println("  Carregado de: $ARQUIVO_MODELO")
        println("  (Para retreinar do zero, mude RETREINAR = true no topo)")
    end
    println("="^60)
else
    RETREINAR  && println("\nRETREINAR = true — retreinando do zero...")
    !RETREINAR && println("\nNenhum modelo encontrado. Iniciando treinamento...")
    params, loss_history = executar_treino()
end

# ===========================
# 8. RESULTADO
# ===========================
β_ap   = exp(params[IDX_log_β])
γ_ap   = exp(params[IDX_log_γ])
R0_est = β_ap / γ_ap

println("\n" * "="^60)
println("  RESULTADO — Dengue Botucatu 2024 (52 semanas)")
println("="^60)
@printf("  log_β armazenado : %.4f\n", params[IDX_log_β])
@printf("  log_γ armazenado : %.4f\n", params[IDX_log_γ])
@printf("  β estimado : %.4f  (= exp(log_β))\n", β_ap)
@printf("  γ estimado : %.4f  (= exp(log_γ))\n", γ_ap)
@printf("  R₀ estimado: %.3f\n", R0_est)
@printf("  Período infeccioso: %.1f semanas\n", 1.0/γ_ap)
println("  Referência Dengue: R₀ ~2–6 | Período infeccioso ~1–2 semanas")
println("="^60)

# ===========================
# 9. PREDIÇÕES FINAIS
# ===========================
t_plot = collect(0.0:0.5:T_FINAL)
preds  = hcat([predict(t, params) for t in t_plot]...)
S_pred = preds[1,:]
I_pred = preds[2,:]
R_pred = preds[3,:]
C_pred = preds[4,:]

ts_rk4, S_rk4, I_rk4, R_rk4, C_rk4 = resolver_rk4(β_ap, γ_ap)

C_pred_abs = C_pred .* C_ESCALA
C_rk4_abs  = C_rk4  .* C_ESCALA

t_inc          = collect(1.0:1.0:52.0)
C_at_t         = [predict(t, params)[4] * C_ESCALA for t in t_inc]
C_prev         = vcat([0.0], C_at_t[1:end-1])
I_semanal_pred = C_at_t .- C_prev

# ===========================
# 10. GRÁFICOS
# ===========================
fs_tick=13; fs_guide=14; fs_title=15; fs_legend=11

n_f1   = EPOCHS_ADAM_F1 ÷ 500
n_lb   = max(0, length(loss_history) - n_f1)

x_f1   = collect(500:500:EPOCHS_ADAM_F1)
x_lb   = collect((EPOCHS_ADAM_F1+500):500:(EPOCHS_ADAM_F1+n_lb*500))
x_hist = vcat(x_f1, x_lb)
n_plot = min(length(x_hist), length(loss_history))

p1 = plot(title="PINN vs Dados Reais — Incidência Semanal (Dengue Botucatu 2024)",
          xlabel="Semana epidemiológica", ylabel="Casos novos",
          titlefontsize=fs_title, guidefontsize=fs_guide,
          tickfontsize=fs_tick, legendfontsize=fs_legend, left_margin=20mm)
scatter!(p1, t_real, I_real_abs,
         label="Dados reais (TABNET)", markersize=5, color=:black)
plot!(p1, t_inc, I_semanal_pred,
      label="PINN ΔC(t)", lw=3, color=:red)

p2 = plot(title="Incidência Cumulativa C(t)  (β=$(round(β_ap,digits=3)) γ=$(round(γ_ap,digits=3)) R₀=$(round(R0_est,digits=2)))",
          xlabel="Semana epidemiológica", ylabel="Casos acumulados",
          titlefontsize=fs_title, guidefontsize=fs_guide,
          tickfontsize=fs_tick, legendfontsize=fs_legend, left_margin=20mm)
scatter!(p2, t_real, C_real_abs,
         label="C real (cumsum TABNET)", markersize=5, color=:black)
plot!(p2, t_plot, C_pred_abs,
      label="PINN C(t)", lw=3, color=:red)
plot!(p2, ts_rk4, C_rk4_abs,
      label="RK4 C(t)", lw=2, ls=:dash, color=:darkred, alpha=0.7)

p3 = plot(x_hist[1:n_plot], loss_history[1:n_plot],
          label="Loss", lw=2, color=:orange, marker=:circle,
          title="Histórico: Adam F1 + L-BFGS (β,γ fixos)",
          xlabel="Iteração", ylabel="Loss", yscale=:log10,
          titlefontsize=fs_title, guidefontsize=fs_guide,
          tickfontsize=fs_tick, legendfontsize=fs_legend, left_margin=20mm)
vline!(p3, [EPOCHS_ADAM_F1],
       label="Adam → L-BFGS", ls=:dash, color=:blue, lw=1.5)

p4 = plot(title="Compartimentos Normalizados — PINN vs RK4",
          xlabel="Semana epidemiológica", ylabel="Proporção da população",
          titlefontsize=fs_title, guidefontsize=fs_guide,
          tickfontsize=fs_tick, legendfontsize=fs_legend, left_margin=20mm)
plot!(p4, t_plot, S_pred, lw=2.5, color=:blue,   label="PINN S(t)")
plot!(p4, t_plot, I_pred, lw=2.5, color=:red,    label="PINN I(t)")
plot!(p4, t_plot, R_pred, lw=2.5, color=:green,  label="PINN R(t)")
plot!(p4, t_plot, C_pred, lw=2.5, color=:purple, label="PINN C(t)")
plot!(p4, ts_rk4, S_rk4, lw=1.5, ls=:dash, color=:darkblue,  label="RK4 S(t)", alpha=0.8)
plot!(p4, ts_rk4, I_rk4, lw=1.5, ls=:dash, color=:darkred,   label="RK4 I(t)", alpha=0.8)
plot!(p4, ts_rk4, R_rk4, lw=1.5, ls=:dash, color=:darkgreen, label="RK4 R(t)", alpha=0.8)
plot!(p4, ts_rk4, C_rk4, lw=1.5, ls=:dash, color=:magenta,   label="RK4 C(t)", alpha=0.8)

pfinal = plot(p1, p2, p3, p4, layout=(4,1), size=(1000,1600), dpi=300)
savefig(pfinal, "sir_pinn_dengue.png")
display(pfinal)
println("\nGráfico salvo em sir_pinn_dengue.png")
