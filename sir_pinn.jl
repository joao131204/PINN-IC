using ForwardDiff
using Random
using Plots
using Plots.PlotMeasures
using Statistics
using Printf
using JLD2
using Optim
using StatsPlots

# ===========================
# 1. PARÂMETROS
# ===========================
RETREINAR = false   # ← mude para false após o primeiro treino

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
I_ESCALA   = maximum(I_real_abs)

t_real  = collect(1.0:1.0:52.0)
T_FINAL = 52.0

PERIODO_INFECCIOSO = 1.0
I0      = (I_real_abs[1] * PERIODO_INFECCIOSO) / C_ESCALA
C0      = 0.0
R0_init = 0.0
S0      = 1.0 - I0 - R0_init

ARQUIVO_PINN = "sir_pinn_fisica.jld2"
ARQUIVO_PURA = "sir_pinn_pura.jld2"

W_IC    = 100.0
W_PHYS  = 50.0
W_DADOS = 50.0

t_collocation = collect(0.0:1.0:T_FINAL)
EPOCHS_ADAM   = 10000
EPOCHS_LBFGS  = 5000

# ===========================
# 2. ARQUITETURA [1→16→32→16→4]
#    N_PARAMS = 1174 (1172 pesos + log_β + log_γ)
# ===========================
const N_PARAMS  = 1174
const N_REDE    = 1172
const IDX_log_β = 1173
const IDX_log_γ = 1174

function inicializar_params(seed=9999)
    Random.seed!(seed)
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
    h1  = tanh.(W1 * t_n .+ b1)
    h2  = tanh.(W2 * h1  .+ b2)
    h3  = tanh.(W3 * h2  .+ b3)
    return σ.(W4 * h3  .+ b4)
end

# ===========================
# 4. RK4
# ===========================
function resolver_rk4(β_val, γ_val)
    dt = 0.1; steps = Int(T_FINAL / dt)
    ts = Float64[]
    Ss, Is, Rs, Cs = Float64[], Float64[], Float64[], Float64[]
    u = [S0, I0, R0_init, C0]
    f(u) = [-β_val*u[1]*u[2], β_val*u[1]*u[2]-γ_val*u[2], γ_val*u[2], β_val*u[1]*u[2]]
    for i in 0:steps
        push!(ts, i*dt); push!(Ss, u[1]); push!(Is, u[2])
        push!(Rs, u[3]); push!(Cs, u[4])
        k1=f(u); k2=f(u.+0.5dt.*k1); k3=f(u.+0.5dt.*k2); k4=f(u.+dt.*k3)
        u = u .+ (dt/6).*(k1.+2k2.+2k3.+k4)
    end
    return ts, Ss, Is, Rs, Cs
end

# ===========================
# 5. RESÍDUOS FÍSICOS
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
function ic_loss(params)
    y0 = predict(0.0, params)
    return W_IC * (
        (y0[1] - S0)^2 + (y0[2] - I0)^2 +
        (y0[3] - R0_init)^2 + (y0[4] - C0)^2
    )
end

function dados_loss(params)
    return W_DADOS * sum(
        (predict(t_real[i], params)[4] - C_real[i])^2
        for i in eachindex(t_real)
    )
end

# PINN — Adam (com física)
loss_pinn_adam(params) =
    ic_loss(params) +
    W_PHYS * sum(compute_residuals(t, params) for t in t_collocation) +
    dados_loss(params)

# PINN — L-BFGS (com física, β e γ fixos)
loss_pinn_lbfgs(params_rede, β_fixo, γ_fixo) =
    ic_loss(params_rede) +
    W_PHYS * sum(compute_residuals_fixo(t, params_rede, β_fixo, γ_fixo) for t in t_collocation) +
    W_DADOS * sum(
        (predict(t_real[i], params_rede)[4] - C_real[i])^2
        for i in eachindex(t_real)
    )

# Rede pura — Adam (sem física)
loss_pura_adam(params) = ic_loss(params) + dados_loss(params)

# Rede pura — L-BFGS (sem física)
loss_pura_lbfgs(params_rede) = ic_loss(params_rede) +
    W_DADOS * sum(
        (predict(t_real[i], params_rede)[4] - C_real[i])^2
        for i in eachindex(t_real)
    )

# ===========================
# 7. TREINO DA PINN (com física)
# ===========================
function treinar_pinn()
    params       = inicializar_params(9999)
    loss_history = Float64[]
    lr              = 0.001
    m_adam          = zeros(N_PARAMS)
    v_adam          = zeros(N_PARAMS)
    beta1, beta2, ε = 0.9, 0.999, 1e-8

    println("\n── PINN | Adam ($(EPOCHS_ADAM) épocas) ──")
    for epoch in 1:EPOCHS_ADAM
        grad = ForwardDiff.gradient(loss_pinn_adam, params)
        m_adam .= beta1 .* m_adam .+ (1-beta1) .* grad
        v_adam .= beta2 .* v_adam .+ (1-beta2) .* (grad.^2)
        m̂ = m_adam ./ (1 - beta1^epoch)
        v̂ = v_adam ./ (1 - beta2^epoch)
        params .-= lr .* m̂ ./ (sqrt.(v̂) .+ ε)

        if epoch % 500 == 0
            l = loss_pinn_adam(params)
            β_e = exp(params[IDX_log_β])
            γ_e = exp(params[IDX_log_γ])
            push!(loss_history, l)
            @printf("[PINN] Epoch %5d | Loss: %.6f | β=%.4f | γ=%.4f | R0=%.3f\n",
                    epoch, l, β_e, γ_e, β_e/γ_e)
        end
    end

    β_fixo = exp(params[IDX_log_β])
    γ_fixo = exp(params[IDX_log_γ])
    println("\n✔ PINN Adam concluída.")
    @printf("  β=%.4f | γ=%.4f | R0=%.3f\n", β_fixo, γ_fixo, β_fixo/γ_fixo)

    @printf("\n── PINN | L-BFGS (%d iter) — β=%.4f γ=%.4f congelados ──\n",
            EPOCHS_LBFGS, β_fixo, γ_fixo)

    params_rede = copy(params[1:N_REDE])
    f_lb(p)     = loss_pinn_lbfgs(p, β_fixo, γ_fixo)
    g_lb!(G, p) = (G .= ForwardDiff.gradient(f_lb, p); G)

    function cb_lb(state)
        if state.iteration > 0 && state.iteration % 500 == 0
            push!(loss_history, state.value)
            @printf("[PINN | L-BFGS] Iter %5d | Loss: %.6f\n",
                    state.iteration, state.value)
        end
        return false
    end

    res = Optim.optimize(f_lb, g_lb!, params_rede, Optim.LBFGS(),
                         Optim.Options(iterations=EPOCHS_LBFGS,
                                       show_trace=false, callback=cb_lb, g_tol=1e-6))
    params[1:N_REDE]  = Optim.minimizer(res)
    params[IDX_log_β] = log(β_fixo)
    params[IDX_log_γ] = log(γ_fixo)

    println("✔ PINN L-BFGS concluído. Convergiu: $(Optim.converged(res))")
    JLD2.jldsave(ARQUIVO_PINN; params=params, loss_history=loss_history)
    println("✔ Salvo em: $ARQUIVO_PINN")
    return params, loss_history
end

# ===========================
# 8. TREINO DA REDE PURA (sem física)
# ===========================
function treinar_pura()
    params       = inicializar_params(1234)   # seed diferente — condições independentes
    loss_history = Float64[]
    lr              = 0.001
    m_adam          = zeros(N_PARAMS)
    v_adam          = zeros(N_PARAMS)
    beta1, beta2, ε = 0.9, 0.999, 1e-8

    println("\n── Rede pura | Adam ($(EPOCHS_ADAM) épocas) ──")
    for epoch in 1:EPOCHS_ADAM
        grad = ForwardDiff.gradient(loss_pura_adam, params)
        m_adam .= beta1 .* m_adam .+ (1-beta1) .* grad
        v_adam .= beta2 .* v_adam .+ (1-beta2) .* (grad.^2)
        m̂ = m_adam ./ (1 - beta1^epoch)
        v̂ = v_adam ./ (1 - beta2^epoch)
        params .-= lr .* m̂ ./ (sqrt.(v̂) .+ ε)

        if epoch % 500 == 0
            l = loss_pura_adam(params)
            push!(loss_history, l)
            @printf("[Pura] Epoch %5d | Loss: %.6f\n", epoch, l)
        end
    end

    println("\n✔ Rede pura Adam concluída.")
    println("\n── Rede pura | L-BFGS ($(EPOCHS_LBFGS) iter) ──")

    params_rede = copy(params[1:N_REDE])
    f_lb(p)     = loss_pura_lbfgs(p)
    g_lb!(G, p) = (G .= ForwardDiff.gradient(f_lb, p); G)

    function cb_lb(state)
        if state.iteration > 0 && state.iteration % 500 == 0
            push!(loss_history, state.value)
            @printf("[Pura | L-BFGS] Iter %5d | Loss: %.6f\n",
                    state.iteration, state.value)
        end
        return false
    end

    res = Optim.optimize(f_lb, g_lb!, params_rede, Optim.LBFGS(),
                         Optim.Options(iterations=EPOCHS_LBFGS,
                                       show_trace=false, callback=cb_lb, g_tol=1e-6))
    params[1:N_REDE] = Optim.minimizer(res)

    println("✔ Rede pura L-BFGS concluído. Convergiu: $(Optim.converged(res))")
    JLD2.jldsave(ARQUIVO_PURA; params=params, loss_history=loss_history)
    println("✔ Salvo em: $ARQUIVO_PURA")
    return params, loss_history
end

# ===========================
# 9. EXECUTAR OU CARREGAR
# ===========================
println("="^60)
println("  SIR-PINN — Comparação: Com Física vs Sem Física")
println("  Dengue Botucatu 2024 | Arquitetura [1→16→32→16→4]")
println("="^60)

if isfile(ARQUIVO_PINN) && !RETREINAR
    println("\nCarregando PINN de $ARQUIVO_PINN...")
    d = JLD2.load(ARQUIVO_PINN)
    params_pinn, hist_pinn = d["params"], d["loss_history"]
else
    params_pinn, hist_pinn = treinar_pinn()
end

if isfile(ARQUIVO_PURA) && !RETREINAR
    println("\nCarregando Rede Pura de $ARQUIVO_PURA...")
    d = JLD2.load(ARQUIVO_PURA)
    params_pura, hist_pura = d["params"], d["loss_history"]
else
    params_pura, hist_pura = treinar_pura()
end

# ===========================
# 10. RESULTADOS
# ===========================
β_ap   = exp(params_pinn[IDX_log_β])
γ_ap   = exp(params_pinn[IDX_log_γ])
R0_est = β_ap / γ_ap

println("\n" * "="^60)
println("  RESULTADO PINN")
println("="^60)
@printf("  β estimado : %.4f\n", β_ap)
@printf("  γ estimado : %.4f\n", γ_ap)
@printf("  R₀ estimado: %.3f\n", R0_est)
@printf("  Período infeccioso: %.1f semanas\n", 1.0/γ_ap)
println("  Referência Dengue: R₀ ~2–6 | Período infeccioso ~1–2 semanas")
println("="^60)

# ===========================
# 11. PREDIÇÕES E MÉTRICAS
# ===========================
t_plot = collect(0.0:0.5:T_FINAL)
t_inc  = collect(1.0:1.0:52.0)

function predicoes(params)
    preds      = hcat([predict(t, params) for t in t_plot]...)
    C_pred_abs = preds[4,:] .* C_ESCALA
    C_at_t     = [predict(t, params)[4] * C_ESCALA for t in t_inc]
    C_prev     = vcat([0.0], C_at_t[1:end-1])
    I_pred     = C_at_t .- C_prev
    return C_pred_abs, I_pred, preds
end

C_pred_pinn, I_pred_pinn, preds_pinn = predicoes(params_pinn)
C_pred_pura, I_pred_pura, preds_pura = predicoes(params_pura)

ts_rk4, S_rk4, I_rk4, R_rk4, C_rk4 = resolver_rk4(β_ap, γ_ap)
C_rk4_abs = C_rk4 .* C_ESCALA

eqm(a, b)  = mean((a .- b).^2)
rmse(a, b) = sqrt(eqm(a, b))

# índices pares de t_plot correspondem a t = 0, 1, 2, ... (passo 0.5, então índice 2 = t=1)
C_pred_pinn_pts = C_pred_pinn[2:2:end]
C_pred_pura_pts = C_pred_pura[2:2:end]

# EQM e RMSE — Incidência semanal
eqm_pinn_I  = eqm(I_pred_pinn,      I_real_abs)
eqm_pura_I  = eqm(I_pred_pura,      I_real_abs)
rmse_pinn_I = rmse(I_pred_pinn,     I_real_abs)
rmse_pura_I = rmse(I_pred_pura,     I_real_abs)

# EQM e RMSE — Cumulativa
eqm_pinn_C  = eqm(C_pred_pinn_pts,  C_real_abs)
eqm_pura_C  = eqm(C_pred_pura_pts,  C_real_abs)
rmse_pinn_C = rmse(C_pred_pinn_pts, C_real_abs)
rmse_pura_C = rmse(C_pred_pura_pts, C_real_abs)

# ===========================
# TABELA DE MÉTRICAS
# ===========================
println("\n" * "="^62)
println("  TABELA DE MÉTRICAS — Incidência Semanal (casos novos)")
println("="^62)
@printf("  %-30s %12s %12s\n", "Métrica", "PINN", "Rede Pura")
println("  " * "-"^58)
@printf("  %-30s %12.2f %12.2f\n", "EQM  (casos²)", eqm_pinn_I,  eqm_pura_I)
@printf("  %-30s %12.2f %12.2f\n", "RMSE (casos)",  rmse_pinn_I, rmse_pura_I)
println("="^62)
println("\n" * "="^62)
println("  TABELA DE MÉTRICAS — Incidência Cumulativa (casos acum.)")
println("="^62)
@printf("  %-30s %12s %12s\n", "Métrica", "PINN", "Rede Pura")
println("  " * "-"^58)
@printf("  %-30s %12.2f %12.2f\n", "EQM  (casos²)", eqm_pinn_C,  eqm_pura_C)
@printf("  %-30s %12.2f %12.2f\n", "RMSE (casos)",  rmse_pinn_C, rmse_pura_C)
println("="^62)
println("  Nota: EQM menor = melhor ajuste aos dados de treino.")
println("  A rede pura tende a ter EQM menor (overfitting).")
println("  A superioridade da PINN se manifesta em generalização.")

# ===========================
# 12. GRÁFICOS
# ===========================
fs_tick=12; fs_guide=13; fs_title=14; fs_legend=10

# p1 — Incidência semanal
p1 = plot(title="Incidência Semanal — PINN vs Rede Pura (Dengue Botucatu 2024)",
          xlabel="Semana epidemiológica", ylabel="Casos novos",
          titlefontsize=fs_title, guidefontsize=fs_guide,
          tickfontsize=fs_tick, legendfontsize=fs_legend, left_margin=20mm)
scatter!(p1, t_real, I_real_abs,
         label="Dados reais (TABNET)", markersize=5, color=:black)
plot!(p1, t_inc, I_pred_pinn,
      label="PINN — com física (RMSE=$(round(rmse_pinn_I, digits=1)))",
      lw=2.5, color=:red)
plot!(p1, t_inc, I_pred_pura,
      label="Rede pura — sem física (RMSE=$(round(rmse_pura_I, digits=1)))",
      lw=2.5, color=:blue, ls=:dash)

# p2 — Cumulativa
p2 = plot(title="Cumulativa C(t) — PINN (β=$(round(β_ap,digits=3)) γ=$(round(γ_ap,digits=3)) R₀=$(round(R0_est,digits=2))) vs Rede Pura",
          xlabel="Semana epidemiológica", ylabel="Casos acumulados",
          titlefontsize=fs_title, guidefontsize=fs_guide,
          tickfontsize=fs_tick, legendfontsize=fs_legend, left_margin=20mm)
scatter!(p2, t_real, C_real_abs,
         label="C real (TABNET)", markersize=5, color=:black)
plot!(p2, t_plot, C_pred_pinn,
      label="PINN C(t) (RMSE=$(round(rmse_pinn_C, digits=1)))",
      lw=2.5, color=:red)
plot!(p2, t_plot, C_pred_pura,
      label="Rede pura C(t) (RMSE=$(round(rmse_pura_C, digits=1)))",
      lw=2.5, color=:blue, ls=:dash)
plot!(p2, ts_rk4, C_rk4_abs,
      label="RK4 C(t)", lw=1.5, ls=:dot, color=:darkred, alpha=0.7)

# p3 — Histórico de loss
n_adam  = EPOCHS_ADAM ÷ 500
n_lb_p  = max(0, length(hist_pinn) - n_adam)
n_lb_r  = max(0, length(hist_pura) - n_adam)
n_lb    = max(n_lb_p, n_lb_r)

x_adam  = collect(500:500:EPOCHS_ADAM)
x_lb    = collect((EPOCHS_ADAM+500):500:(EPOCHS_ADAM + n_lb*500))
x_hist  = vcat(x_adam, x_lb)
n_plot  = min(length(x_hist), length(hist_pinn), length(hist_pura))

p3 = plot(title="Histórico de Loss (log) — Adam + L-BFGS",
          xlabel="Iteração", ylabel="Loss", yscale=:log10,
          titlefontsize=fs_title, guidefontsize=fs_guide,
          tickfontsize=fs_tick, legendfontsize=fs_legend, left_margin=20mm)
plot!(p3, x_hist[1:n_plot], hist_pinn[1:n_plot],
      label="PINN (com física)", lw=2, color=:red, marker=:circle, markersize=3)
plot!(p3, x_hist[1:n_plot], hist_pura[1:n_plot],
      label="Rede pura (sem física)", lw=2, color=:blue, marker=:circle, markersize=3)
vline!(p3, [EPOCHS_ADAM], label="Adam → L-BFGS", ls=:dash, color=:gray, lw=1.5)

# p4 — Compartimentos PINN vs RK4
p4 = plot(title="Compartimentos PINN vs RK4 (β=$(round(β_ap,digits=3)) γ=$(round(γ_ap,digits=3)))",
          xlabel="Semana epidemiológica", ylabel="Proporção da população",
          titlefontsize=fs_title, guidefontsize=fs_guide,
          tickfontsize=fs_tick, legendfontsize=fs_legend, left_margin=20mm)
plot!(p4, t_plot, preds_pinn[1,:], lw=2.5, color=:blue,   label="PINN S(t)")
plot!(p4, t_plot, preds_pinn[2,:], lw=2.5, color=:red,    label="PINN I(t)")
plot!(p4, t_plot, preds_pinn[3,:], lw=2.5, color=:green,  label="PINN R(t)")
plot!(p4, t_plot, preds_pinn[4,:], lw=2.5, color=:purple, label="PINN C(t)")
plot!(p4, ts_rk4, S_rk4, lw=1.5, ls=:dash, color=:darkblue,  label="RK4 S(t)", alpha=0.8)
plot!(p4, ts_rk4, I_rk4, lw=1.5, ls=:dash, color=:darkred,   label="RK4 I(t)", alpha=0.8)
plot!(p4, ts_rk4, R_rk4, lw=1.5, ls=:dash, color=:darkgreen, label="RK4 R(t)", alpha=0.8)
plot!(p4, ts_rk4, C_rk4, lw=1.5, ls=:dash, color=:magenta,   label="RK4 C(t)", alpha=0.8)

# p5 — Comparação de EQM em barras (Incidência e Cumulativa)
categorias = ["Incidência\nsemanal", "Cumulativa"]
eqm_pinn_vals = [eqm_pinn_I, eqm_pinn_C]
eqm_pura_vals = [eqm_pura_I, eqm_pura_C]

p5 = groupedbar(
    categorias,
    hcat(eqm_pinn_vals, eqm_pura_vals),
    label=["PINN (com física)" "Rede pura (sem física)"],
    color=[:red :blue],
    alpha=0.75,
    title="Comparação de EQM — PINN vs Rede Pura",
    ylabel="EQM (casos²)",
    titlefontsize=fs_title, guidefontsize=fs_guide,
    tickfontsize=fs_tick, legendfontsize=fs_legend,
    left_margin=20mm, bar_width=0.6
)

pfinal = plot(p1, p2, p3, p4, p5, layout=(5,1), size=(1000,2000), dpi=300)
savefig(pfinal, "sir_pinn_comparacao.png")
display(pfinal)
println("\nGráfico salvo em sir_pinn_comparacao.png")