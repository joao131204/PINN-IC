using ForwardDiff
using Random
using Plots
using Plots.PlotMeasures
using StatsPlots
using Statistics
using Dates
using Printf
using JLD2
using Optim

# ===========================
# 1. PARÂMETROS
# ===========================
# VERSÃO v4.3-positividade-hard (baseada na v4.2, que testou uma
# penalidade SOFT de monotonicidade em C(t) e teve resultado misto:
# melhorou o ajuste no treino, mas piorou a generalização na projeção
# — RMSE de projeção subiu de 6.723,84 para 7.121,59 — e ainda restaram
# 82 pontos de violação de monotonicidade fora da malha de colocação
# fiscalizada, porque a penalidade soft só age exatamente nos 210
# pontos inteiros checados, não entre eles).
#
# MUDANÇA NESTA VERSÃO: em vez de PEDIR (via penalidade) que dC/dt seja
# não-negativo, a arquitetura agora GARANTE isso por construção — uma
# restrição "hard" de positividade, seguindo a literatura de redes
# monotônicas/positivas (não é uma técnica específica de PINN, é um
# tópico geral de arquitetura de redes neurais, como sugerido pelo
# professor):
#
#   - Wehenkel & Louppe, "Unconstrained Monotonic Neural Networks"
#     (arXiv:1908.05164): uma função é monotônica sempre que sua
#     derivada é estritamente positiva; isso pode ser garantido com uma
#     rede livre cuja ÚNICA restrição é a positividade da sua saída — e
#     a função em si é obtida INTEGRANDO essa saída positiva, nunca
#     diferenciando um estado livre.
#   - PIDL para produção de entropia (arXiv:2606.01179): mostra
#     empiricamente exatamente o padrão que vimos aqui — restrição hard
#     via softplus deu 0 violações em 100% dos pontos testados; a
#     penalidade soft equivalente ainda produziu violações perto de
#     regiões com poucos dados (no nosso caso, a região de projeção).
#   - PINNs para SEIR/SEIRD (arXiv:2509.22760, arXiv:2605.19886):
#     confirmam que usar softplus na camada de saída para garantir
#     não-negatividade já é prática padrão especificamente em modelos
#     compartimentais epidemiológicos — a mesma família do nosso SIR.
#   - "Constrained Monotonic Neural Networks" (arXiv:2205.11775) e o
#     survey arXiv:2505.02537: dão o enquadramento geral "soft vs. hard
#     monotonicity" — penalidades só garantem a propriedade dentro da
#     distribuição amostrada; restrições arquiteturais garantem em
#     qualquer ponto do domínio, por construção.
#
# COMO ISSO FOI IMPLEMENTADO AQUI:
#   A rede continua livre para aprender que TAXA de novos casos ela
#   acha melhor a cada instante — só que agora a 4ª saída da rede
#   (antes interpretada como "C(t)", passando por sigmoide) passa a ser
#   interpretada como "dC/dt(t)" (a taxa instantânea), passando por uma
#   ativação softplus (sempre >= 0, nunca negativa — em QUALQUER ponto
#   real de t, não só nos pontos de colocação fiscalizados). C(t) deixa
#   de ser uma saída direta da rede e passa a ser obtida por integração
#   (soma cumulativa) dessa taxa, a partir de C(0) = 0.
#
#   Isso é o "Passo A" — mais contido que a ideia maior discutida
#   antes (definir a taxa de C como sendo LITERALMENTE β(t)·S(t)·I(t),
#   sem nenhum parâmetro livre de rede para isso, o "Passo B"). Aqui a
#   rede ainda tem liberdade para aprender a taxa; o resíduo físico
#   continua *sugerindo* que essa taxa deveria bater com β·S·I, de
#   forma soft, igual antes — só a garantia de SINAL (nunca negativo)
#   que virou hard/arquitetural.
#
#   BÔNUS: como paramos de diferenciar uma sigmoide multiplicada por
#   C_ESCALA para obter a incidência, a penalidade de monotonicidade da
#   v4.2 (W_MONO / mono_loss) deixou de ser necessária e foi removida —
#   a garantia agora é estrutural, não custa nada na função de perda.
#
# RETREINAR = false: procura pesos já treinados em
# sir_pinn_fisica_v4_3_positividade_hard.jld2 /
# sir_pinn_pura_v4_1_clima_suave.jld2 (a Rede Pura é reaproveitada da
# v4.1 — nada muda para ela aqui) na mesma pasta do script.
#
# ATENÇÃO: a arquitetura de saída mudou (a 4ª saída agora é uma TAXA,
# não mais um estado C∈(0,1)). Isso significa que os pesos salvos da
# PINN em versões anteriores (v4.1/v4.2) NÃO são compatíveis com esta
# versão — é necessário treinar do zero (RETREINAR efetivamente vai
# acontecer na primeira vez que rodar isso, já que o arquivo .jld2 novo
# ainda não existe). Espere as ~12h de treino novamente para a PINN.
RETREINAR = false

# -----------------------------------------------------------------------
# DADOS DE TREINO — Malária Amazonas 2022+2023+2024 (156 semanas)
# Fonte: TABNET / SINAN — semana epidemiológica (convenção SINAN/CDC)
# -----------------------------------------------------------------------
I_treino_abs = Float64[
    # --- ANO 2022 (52 semanas) ---
    819, 753, 753, 771, 745, 833, 946, 796, 811, 854, 877, 849,
    654, 776, 665, 674, 672, 649, 752, 858, 753, 737, 962, 982,
    895, 692, 947, 1132, 1196, 1000, 1147, 1221, 1203, 1205, 1140, 1262,
    1272, 1357, 1221, 1248, 1075, 950, 859, 777, 884, 890, 801, 769,
    793, 882, 746, 729,

    # --- ANO 2023 (52 semanas) ---
    850, 809, 821, 872, 699, 626, 688, 520, 549, 581, 595, 610,
    494, 507, 700, 613, 708, 562, 674, 729, 768, 885, 969, 1138,
    1105, 1098, 1092, 1529, 1629, 1660, 1727, 1987, 1573, 1389, 1504, 1643,
    1399, 1193, 975, 1051, 965, 1132, 854, 763, 932, 1042, 963, 891,
    1082, 1007, 1040, 805,

    # --- ANO 2024 (52 semanas) ---
    1161, 959, 1022, 888, 846, 940, 904, 956, 922, 912,
    886, 962, 694, 977, 1057, 972, 1050, 930, 1264, 1102,
    1192, 1050, 1167, 1093, 1062, 1154, 1225, 1141, 1243, 1303,
    1285, 1389, 1540, 1320, 1118, 1196, 1254, 1101, 922, 913,
    963, 724, 928, 871, 1028, 823, 939, 1015, 996, 1024, 884,
    807
]

# -----------------------------------------------------------------------
# DADOS DE PROJEÇÃO — Malária Amazonas 2025 (53 semanas)
# Fonte: TABNET / SINAN
# Estes dados NUNCA entram na função de perda. Servem só de gabarito,
# depois do treino, para comparar com a projeção que a rede faz para o
# futuro.
# -----------------------------------------------------------------------
I_teste_abs = Float64[
    570, 969, 942, 926, 917, 956, 894, 891, 801, 803,
    833, 780, 784, 728, 804, 743, 800, 728, 986, 1021,
    1039, 985, 1055, 1237, 1112, 1204, 1231, 1589, 1236, 1132,
    1421, 1697, 1578, 1435, 1468, 1429, 1430, 1243, 1240, 1342,
    1341, 1201, 1105, 976, 1050, 907, 872, 851, 840, 857,
    920, 751, 496
]

N_TREINO = length(I_treino_abs)   # 156
N_TESTE  = length(I_teste_abs)    # 53
T_FINAL  = Float64(N_TREINO)      # 156.0 — fim do treino / início da projeção

T_HORIZON = T_FINAL + Float64(N_TESTE)   # 209.0

# =========================================================================
# CLIMA COMO ENTRADA EXTRA DA REDE (chuva, temperatura, umidade — atraso
# POR VARIÁVEL, calibrado por correlação cruzada com os 3 anos de dados)
# =========================================================================
ARQUIVO_CLIMA = joinpath(@__DIR__, "amazonas_dados_diarios_2022_2024.csv")
ANOS_TREINO   = [2022, 2023, 2024]

ATRASO_CHUVA = 2  # semanas
ATRASO_TEMP  = 0  # semanas
ATRASO_UMID  = 0  # semanas

function ler_clima_diario(caminho)
    chuva_dia = Dict{Date, Vector{Float64}}()
    temp_dia  = Dict{Date, Vector{Float64}}()
    umid_dia  = Dict{Date, Vector{Float64}}()

    for linha in readlines(caminho)[2:end]
        campos = split(strip(linha), ';')
        length(campos) < 6 && continue
        try
            data = Date(strip(campos[1]), dateformat"dd/mm/yyyy")
            if strip(campos[4]) != ""
                push!(get!(chuva_dia, data, Float64[]), parse(Float64, campos[4]))
            end
            if strip(campos[5]) != ""
                push!(get!(temp_dia, data, Float64[]), parse(Float64, campos[5]))
            end
            if strip(campos[6]) != ""
                push!(get!(umid_dia, data, Float64[]), parse(Float64, campos[6]))
            end
        catch
            continue
        end
    end

    chuva_diaria = Dict(d => mean(v) for (d, v) in chuva_dia)
    temp_diaria  = Dict(d => mean(v) for (d, v) in temp_dia)
    umid_diaria  = Dict(d => mean(v) for (d, v) in umid_dia)
    return chuva_diaria, temp_diaria, umid_diaria
end

chuva_diaria, temp_diaria, umid_diaria = ler_clima_diario(ARQUIVO_CLIMA)
println("Dias com leitura de chuva: $(length(chuva_diaria)) | temp: $(length(temp_diaria)) | umidade: $(length(umid_diaria))")

function domingo_da_semana1(ano::Int)
    jan4 = Date(ano, 1, 4)
    offset_domingo = mod(Dates.dayofweek(jan4), 7)
    return jan4 - Day(offset_domingo)
end

function semana_epidemiologica(d::Date)
    ano = Dates.year(d)
    dom_atual = domingo_da_semana1(ano)
    dom_prox  = domingo_da_semana1(ano + 1)
    if d >= dom_prox
        semana = div(Dates.value(d - dom_prox), 7) + 1
        return (ano + 1, semana)
    elseif d < dom_atual
        dom_ant = domingo_da_semana1(ano - 1)
        semana = div(Dates.value(d - dom_ant), 7) + 1
        return (ano - 1, semana)
    else
        semana = div(Dates.value(d - dom_atual), 7) + 1
        return (ano, semana)
    end
end

function agregar_clima_semanal(chuva_diaria, temp_diaria, umid_diaria)
    todas_datas = sort(collect(union(keys(chuva_diaria), keys(temp_diaria), keys(umid_diaria))))
    grupos = Dict{Tuple{Int,Int}, Vector{Date}}()
    for d in todas_datas
        chave = semana_epidemiologica(d)
        push!(get!(grupos, chave, Date[]), d)
    end

    chuva_sem = Dict{Tuple{Int,Int}, Float64}()
    temp_sem  = Dict{Tuple{Int,Int}, Float64}()
    umid_sem  = Dict{Tuple{Int,Int}, Float64}()
    for (chave, dias) in grupos
        length(dias) != 7 && continue
        chuva_vals = [chuva_diaria[d] for d in dias if haskey(chuva_diaria, d)]
        temp_vals  = [temp_diaria[d]  for d in dias if haskey(temp_diaria, d)]
        umid_vals  = [umid_diaria[d]  for d in dias if haskey(umid_diaria, d)]
        isempty(chuva_vals) && continue
        chuva_sem[chave] = sum(chuva_vals)
        !isempty(temp_vals) && (temp_sem[chave] = mean(temp_vals))
        !isempty(umid_vals) && (umid_sem[chave] = mean(umid_vals))
    end
    return chuva_sem, temp_sem, umid_sem
end

chuva_sem, temp_sem, umid_sem = agregar_clima_semanal(chuva_diaria, temp_diaria, umid_diaria)
println("Semanas epidemiológicas completas agregadas: $(length(chuva_sem))")

default_chuva = mean(collect(values(chuva_sem)))
default_temp  = mean(collect(values(temp_sem)))
default_umid  = mean(collect(values(umid_sem)))

function t_para_ano_semana(t::Int)
    ano    = ANOS_TREINO[1] + div(t - 1, 52)
    semana = mod(t - 1, 52) + 1
    return ano, semana
end

function clima_real_em(t_idx::Int, lag::Int, dic_sem, default::Float64)
    tt = t_idx - lag
    if tt < 1 || tt > N_TREINO
        return default
    end
    ano, semana = t_para_ano_semana(tt)
    return get(dic_sem, (ano, semana), default)
end

clima_lag_chuva = [clima_real_em(t, ATRASO_CHUVA, chuva_sem, default_chuva) for t in 0:N_TREINO]
clima_lag_temp  = [clima_real_em(t, ATRASO_TEMP,  temp_sem,  default_temp)  for t in 0:N_TREINO]
clima_lag_umid  = [clima_real_em(t, ATRASO_UMID,  umid_sem,  default_umid) for t in 0:N_TREINO]

μ_chuva, σ_chuva = mean(clima_lag_chuva[2:end]), std(clima_lag_chuva[2:end])
μ_temp,  σ_temp  = mean(clima_lag_temp[2:end]),  std(clima_lag_temp[2:end])
μ_umid,  σ_umid  = mean(clima_lag_umid[2:end]),  std(clima_lag_umid[2:end])

norm_chuva(x) = (x - μ_chuva) / σ_chuva
norm_temp(x)  = (x - μ_temp)  / σ_temp
norm_umid(x)  = (x - μ_umid)  / σ_umid

function montar_tabela_clima()
    tabela = zeros(3, Int(T_HORIZON) + 1)
    for t in 0:Int(T_HORIZON)
        if t <= N_TREINO
            c, tp, u = clima_lag_chuva[t + 1], clima_lag_temp[t + 1], clima_lag_umid[t + 1]
        else
            t_fonte = t - 52
            while t_fonte > N_TREINO
                t_fonte -= 52
            end
            t_fonte = max(t_fonte, 0)
            c, tp, u = clima_lag_chuva[t_fonte + 1], clima_lag_temp[t_fonte + 1], clima_lag_umid[t_fonte + 1]
        end
        tabela[1, t + 1] = norm_chuva(c)
        tabela[2, t + 1] = norm_temp(tp)
        tabela[3, t + 1] = norm_umid(u)
    end
    return tabela
end

const TABELA_CLIMA = montar_tabela_clima()

function clima_em(t)
    # Interpolação linear entre os pontos semanais. Só usa o valor
    # primal de t (sem propagar derivada) — evita diferenciar clima(t)
    # dentro dos resíduos físicos via ForwardDiff.
    t_val = t isa ForwardDiff.Dual ? ForwardDiff.value(t) : t
    t_val = clamp(t_val, 0.0, T_HORIZON)
    idx_lo = clamp(floor(Int, t_val), 0, Int(T_HORIZON) - 1)
    idx_hi = idx_lo + 1
    frac   = t_val - idx_lo
    return TABELA_CLIMA[:, idx_lo + 1] .* (1 - frac) .+ TABELA_CLIMA[:, idx_hi + 1] .* frac
end

C_treino_abs = cumsum(I_treino_abs)

FATOR_SEGURANCA_ESCALA = 3.0
C_ESCALA = FATOR_SEGURANCA_ESCALA * maximum(C_treino_abs)

C_treino = C_treino_abs ./ C_ESCALA

t_real = collect(1.0:1.0:T_FINAL)

PERIODO_INFECCIOSO = 1.0
I0      = (I_treino_abs[1] * PERIODO_INFECCIOSO) / C_ESCALA
C0      = 0.0
R0_init = 0.0
S0      = 1.0 - I0 - R0_init

t_teste_abs = collect((T_FINAL + 1.0):1.0:(T_FINAL + Float64(N_TESTE)))
C_teste_abs = C_treino_abs[end] .+ cumsum(I_teste_abs)

ARQUIVO_PINN = joinpath(@__DIR__, "sir_pinn_fisica_v4_3_positividade_hard.jld2")
ARQUIVO_PURA = joinpath(@__DIR__, "sir_pinn_pura_v4_1_clima_suave.jld2")  # reaproveitada da v4.1 (nada muda para a Rede Pura)

W_IC    = 100.0
W_PHYS  = 50.0
W_DADOS = 50.0
# W_MONO removido nesta versão — a monotonicidade agora é uma garantia
# arquitetural (hard), não uma penalidade (soft). Ver comentário no topo.

t_collocation = collect(0.0:1.0:T_HORIZON)   # 0..209 — universo COMPLETO, sem amostragem
                                              # (span cobre treino E projeção ao mesmo tempo)

EPOCHS_ADAM   = 10000
EPOCHS_LBFGS  = 5000

# ===========================
# 2. ARQUITETURA [4→16→32→16→4]
#    N_PARAMS = 1225 (1220 pesos + log_β₀ + c_chuva + c_temp + c_umid +
#    log_γ). β(t) é função do clima; γ continua constante.
#    A 4ª saída da rede agora representa dC/dt (taxa), não mais C.
# ===========================
const N_ENTRADA   = 4
const N_PARAMS    = 1225
const N_REDE      = 1220
const IDX_log_β0   = 1221
const IDX_c_chuva  = 1222
const IDX_c_temp   = 1223
const IDX_c_umid   = 1224
const IDX_log_γ    = 1225

function inicializar_params(seed=9999)
    Random.seed!(seed)
    params = zeros(Float64, N_PARAMS)
    offset = 0

    lim = sqrt(6.0 / (N_ENTRADA + 16))
    n = 16*N_ENTRADA;  params[offset+1:offset+n] = (rand(n).*2lim).-lim;  offset += n
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

    # A 4ª bias começa bem negativa de propósito: softplus(-5) ≈ 0.0067,
    # ou seja, a taxa inicial já nasce pequena (evita que a rede comece
    # "despejando" casos demais antes mesmo de ver os dados).
    params[offset] = -5.0   # último elemento de b4, referente à 4ª saída

    params[IDX_log_β0]  = log(1.5)
    params[IDX_c_chuva] = 0.0
    params[IDX_c_temp]  = 0.0
    params[IDX_c_umid]  = 0.0
    params[IDX_log_γ]   = log(0.2)
    return params
end

function beta_em(t, log_β0, c_chuva, c_temp, c_umid)
    clima = clima_em(t)
    return exp(log_β0 + c_chuva*clima[1] + c_temp*clima[2] + c_umid*clima[3])
end

# ===========================
# 3. FORWARD PASS
# ===========================
σ(x) = 1.0 / (1.0 + exp(-x))

# Softplus numericamente estável (evita overflow de exp para x grande).
# Garante saída sempre >= 0, para QUALQUER valor real de entrada — essa
# é a restrição HARD de positividade (Wehenkel & Louppe, 2019; e o
# padrão usado em PINNs de SEIR/SEIRD e no exemplo de entropia citados
# no topo do arquivo).
softplus(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

function predict(t::T, params) where T
    offset = 0
    W1 = reshape(params[offset+1:offset+16*N_ENTRADA], 16, N_ENTRADA);  offset += 16*N_ENTRADA
    b1 =         params[offset+1:offset+16];             offset += 16
    W2 = reshape(params[offset+1:offset+512], 32, 16);  offset += 512
    b2 =         params[offset+1:offset+32];             offset += 32
    W3 = reshape(params[offset+1:offset+512], 16, 32);  offset += 512
    b3 =         params[offset+1:offset+16];             offset += 16
    W4 = reshape(params[offset+1:offset+64],   4, 16);  offset += 64
    b4 =         params[offset+1:offset+4]

    clima = clima_em(t)
    t_n   = vcat(t / T_HORIZON, clima)
    h1  = tanh.(W1 * t_n .+ b1)
    h2  = tanh.(W2 * h1  .+ b2)
    h3  = tanh.(W3 * h2  .+ b3)
    z   = W4 * h3 .+ b4   # 4 valores "crus" (pré-ativação)

    S = σ(z[1])
    I = σ(z[2])
    R = σ(z[3])
    taxa_C = softplus(z[4])   # <- MUDANÇA: era σ(z[4]) interpretado como C; agora é a
                              #    TAXA dC/dt, garantida >= 0 por construção, sempre.
    return [S, I, R, taxa_C]
end

# ===========================
# 3c. PREDICT DA REDE PURA (arquitetura ANTIGA — sigmoide nas 4 saídas)
# ===========================
# CORREÇÃO IMPORTANTE: a Rede Pura foi treinada na v4.1 com a 4ª saída
# sendo C(t) diretamente via sigmoide (valor sempre em (0,1)). Ela NÃO
# pode reutilizar a predict() de cima, que agora aplica softplus na 4ª
# saída (semântica de TAXA) — softplus e sigmoide produzem números
# completamente diferentes para os mesmos pesos "crus", então usar a
# predict() nova nos pesos antigos da Pura geraria uma "C(t)" sem
# nenhum sentido (foi exatamente esse bug que gerou os números
# inflados de Rede Pura na rodada anterior — RMSE de projeção "subindo"
# para 15335,88 não era um resultado real, era esse erro).
# Esta função replica exatamente a arquitetura da v4.1/v4.2, mantendo a
# Rede Pura como um baseline consistente e comparável entre versões.
function predict_pura(t::T, params) where T
    offset = 0
    W1 = reshape(params[offset+1:offset+16*N_ENTRADA], 16, N_ENTRADA);  offset += 16*N_ENTRADA
    b1 =         params[offset+1:offset+16];             offset += 16
    W2 = reshape(params[offset+1:offset+512], 32, 16);  offset += 512
    b2 =         params[offset+1:offset+32];             offset += 32
    W3 = reshape(params[offset+1:offset+512], 16, 32);  offset += 512
    b3 =         params[offset+1:offset+16];             offset += 16
    W4 = reshape(params[offset+1:offset+64],   4, 16);  offset += 64
    b4 =         params[offset+1:offset+4]

    clima = clima_em(t)
    t_n   = vcat(t / T_HORIZON, clima)
    h1  = tanh.(W1 * t_n .+ b1)
    h2  = tanh.(W2 * h1  .+ b2)
    h3  = tanh.(W3 * h2  .+ b3)
    return σ.(W4 * h3 .+ b4)   # todas as 4 saídas via sigmoide, igual v4.1/v4.2 — 4ª saída = C(t) direto
end

# ===========================
# 3b. RECONSTRUÇÃO DE C(t) POR INTEGRAÇÃO  [NOVO — restrição hard]
# ===========================
# C(t) não é mais uma saída livre da rede: é obtida integrando a taxa
# (garantidamente >= 0) a partir de C(0) = 0. Isso é o que torna C(t)
# monotonicamente não-decrescente por CONSTRUÇÃO, em qualquer ponto —
# não uma penalidade que só age nos pontos fiscalizados.
#
# t_collocation já cobre o universo COMPLETO (0..209, treino+projeção)
# com passo 1 semana, então uma única soma cumulativa cobre os dois
# períodos de uma vez.
function C_grid_completo(params)
    taxas = [predict(t, params)[4] for t in t_collocation]
    # soma cumulativa "exclusiva": C_grid[i] = soma das taxas ANTES do
    # ponto i (regra do retângulo à esquerda, dt=1 semana). Isso garante
    # C_grid[1] (t=0) = 0.0 EXATAMENTE, sem precisar de termo na perda.
    return cumsum(taxas) .- taxas
end

# Versão genérica para malhas mais finas (usada só para plotar), dado
# um ponto de partida C_inicio e um passo dt_local uniforme.
function C_em_malha(ts, params, C_inicio)
    dt_local = ts[2] - ts[1]
    taxas = [predict(t, params)[4] for t in ts]
    incrementos = (cumsum(taxas) .- taxas) .* dt_local
    return incrementos .+ C_inicio
end

# ===========================
# 4. RK4 (β(t) variando com o clima) — inalterado
# ===========================
# Esta função resolve o SIR clássico via Runge-Kutta 4ª ordem, de forma
# totalmente independente da rede — serve só de verificação/comparação
# visual. dC/dt aqui já sempre foi β·S·I (sempre >= 0), então nada
# muda aqui.
function resolver_rk4(log_β0, c_chuva, c_temp, c_umid, γ_val, t_final)
    dt = 0.1; steps = Int(t_final / dt)
    ts = Float64[]
    Ss, Is, Rs, Cs = Float64[], Float64[], Float64[], Float64[]
    u = [S0, I0, R0_init, C0]
    function f(u, t)
        β_t = beta_em(t, log_β0, c_chuva, c_temp, c_umid)
        return [-β_t*u[1]*u[2], β_t*u[1]*u[2]-γ_val*u[2], γ_val*u[2], β_t*u[1]*u[2]]
    end
    for i in 0:steps
        t = i*dt
        push!(ts, t); push!(Ss, u[1]); push!(Is, u[2])
        push!(Rs, u[3]); push!(Cs, u[4])
        k1=f(u,t); k2=f(u.+0.5dt.*k1, t+0.5dt); k3=f(u.+0.5dt.*k2, t+0.5dt); k4=f(u.+dt.*k3, t+dt)
        u = u .+ (dt/6).*(k1.+2k2.+2k3.+k4)
    end
    return ts, Ss, Is, Rs, Cs
end

# ===========================
# 5. RESÍDUOS FÍSICOS (SIR, β(t) modulado por clima)
# ===========================
# MUDANÇA: res_C não usa mais ForwardDiff.derivative (não precisamos
# diferenciar C, já que a rede agora entrega a taxa diretamente). O
# resíduo compara a taxa prevista pela rede com o valor teórico β·S·I —
# mesma ideia de antes, só que sem a etapa de diferenciação.
function compute_residuals(t, params)
    β_val = beta_em(t, params[IDX_log_β0], params[IDX_c_chuva], params[IDX_c_temp], params[IDX_c_umid])
    γ_val = exp(params[IDX_log_γ])
    y = predict(t, params)
    S, I, R, taxa_C = y[1], y[2], y[3], y[4]
    derivs = ForwardDiff.derivative(τ -> predict(τ, params), t)
    dS, dI, dR = derivs[1], derivs[2], derivs[3]   # a 4ª componente de derivs não é usada (seria d(taxa)/dt, irrelevante aqui)
    res_S = dS - (-β_val * S * I)
    res_I = dI - ( β_val * S * I - γ_val * I)
    res_R = dR - ( γ_val * I)
    res_C = taxa_C - ( β_val * S * I)   # <- comparação direta, sem derivar
    return res_S^2 + res_I^2 + res_R^2 + res_C^2
end

function compute_residuals_fixo(t, params_rede, log_β0_fixo, c_chuva_fixo, c_temp_fixo, c_umid_fixo, γ_fixo)
    β_val = beta_em(t, log_β0_fixo, c_chuva_fixo, c_temp_fixo, c_umid_fixo)
    y = predict(t, params_rede)
    S, I, R, taxa_C = y[1], y[2], y[3], y[4]
    derivs = ForwardDiff.derivative(τ -> predict(τ, params_rede), t)
    dS, dI, dR = derivs[1], derivs[2], derivs[3]
    res_S = dS - (-β_val * S * I)
    res_I = dI - ( β_val * S * I - γ_fixo * I)
    res_R = dR - ( γ_fixo * I)
    res_C = taxa_C - ( β_val * S * I)
    return res_S^2 + res_I^2 + res_R^2 + res_C^2
end

# ===========================
# 6. FUNÇÕES DE PERDA
# ===========================
function ic_loss(params)
    y0 = predict(0.0, params)
    # Note: o termo para C(0) foi REMOVIDO — C(0)=0 já é garantido
    # exatamente pela construção de C_grid_completo (soma cumulativa
    # exclusiva começando em zero), não precisa entrar na perda.
    return W_IC * (
        (y0[1] - S0)^2 + (y0[2] - I0)^2 +
        (y0[3] - R0_init)^2
    )
end

function dados_loss(params)
    C_grid = C_grid_completo(params)
    # t_collocation[k+1] == t_real[k], já que t_collocation = 0,1,2,...
    # com passo 1 (t_real = 1,2,...,156 é um subconjunto exato)
    return W_DADOS * sum(
        (C_grid[k+1] - C_treino[k])^2
        for k in eachindex(t_real)
    )
end

loss_pinn_adam(params) =
    ic_loss(params) +
    W_PHYS * sum(compute_residuals(t, params) for t in t_collocation) +
    dados_loss(params)
    # mono_loss removido — a garantia agora é arquitetural (hard)

loss_pinn_lbfgs(params_rede, log_β0_fixo, c_chuva_fixo, c_temp_fixo, c_umid_fixo, γ_fixo) =
    ic_loss(params_rede) +
    W_PHYS * sum(compute_residuals_fixo(t, params_rede, log_β0_fixo, c_chuva_fixo, c_temp_fixo, c_umid_fixo, γ_fixo) for t in t_collocation) +
    dados_loss(params_rede)

# CORREÇÃO: usam predict_pura (sigmoide) e uma ic_loss/dados_loss
# próprias, com o termo de C(0) restaurado — porque, na arquitetura
# antiga, C(0)=0 NÃO é garantido por construção (é uma sigmoide livre),
# precisa do termo soft na perda para ser empurrado pra perto de zero.
function ic_loss_pura(params)
    y0 = predict_pura(0.0, params)
    return W_IC * (
        (y0[1] - S0)^2 + (y0[2] - I0)^2 +
        (y0[3] - R0_init)^2 + (y0[4] - C0)^2
    )
end

function dados_loss_pura(params)
    return W_DADOS * sum(
        (predict_pura(t_real[i], params)[4] - C_treino[i])^2
        for i in eachindex(t_real)
    )
end

loss_pura_adam(params) = ic_loss_pura(params) + dados_loss_pura(params)   # Rede Pura: sem física (baseline)
                                                                            # Ela é o baseline "zero conhecimento" e
                                                                            # não precisa da mudança de positividade
                                                                            # (nunca teve o problema de violação de
                                                                            # monotonicidade: 0 violações mesmo sem
                                                                            # nenhuma restrição, como já vimos).

loss_pura_lbfgs(params_rede) = ic_loss_pura(params_rede) +
    W_DADOS * sum(
        (predict_pura(t_real[i], params_rede)[4] - C_treino[i])^2
        for i in eachindex(t_real)
    )

# ===========================
# 7. TREINO DA PINN (com física + positividade hard)
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
            γ_e  = exp(params[IDX_log_γ])
            β0_e = exp(params[IDX_log_β0])
            push!(loss_history, l)
            @printf("[PINN] Epoch %5d | Loss: %.6f | β0=%.4f | c_chuva=%.4f | c_temp=%.4f | c_umid=%.4f | γ=%.4f\n",
                    epoch, l, β0_e, params[IDX_c_chuva], params[IDX_c_temp], params[IDX_c_umid], γ_e)
        end
    end

    log_β0_fixo  = params[IDX_log_β0]
    c_chuva_fixo = params[IDX_c_chuva]
    c_temp_fixo  = params[IDX_c_temp]
    c_umid_fixo  = params[IDX_c_umid]
    γ_fixo       = exp(params[IDX_log_γ])
    println("\n✔ PINN Adam concluída.")
    @printf("  β0=%.4f | c_chuva=%.4f | c_temp=%.4f | c_umid=%.4f | γ=%.4f\n",
            exp(log_β0_fixo), c_chuva_fixo, c_temp_fixo, c_umid_fixo, γ_fixo)
    @printf("\n── PINN | L-BFGS (%d iter) — coeficientes de β(t) e γ congelados ──\n", EPOCHS_LBFGS)

    params_rede = copy(params[1:N_REDE])
    f_lb(p)     = loss_pinn_lbfgs(p, log_β0_fixo, c_chuva_fixo, c_temp_fixo, c_umid_fixo, γ_fixo)
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
    params[1:N_REDE]   = Optim.minimizer(res)
    params[IDX_log_β0]  = log_β0_fixo
    params[IDX_c_chuva] = c_chuva_fixo
    params[IDX_c_temp]  = c_temp_fixo
    params[IDX_c_umid]  = c_umid_fixo
    params[IDX_log_γ]   = log(γ_fixo)

    println("✔ PINN L-BFGS concluído. Convergiu: $(Optim.converged(res))")
    JLD2.jldsave(ARQUIVO_PINN; params=params, loss_history=loss_history)
    println("✔ Salvo em: $ARQUIVO_PINN")
    return params, loss_history
end

# ===========================
# 8. TREINO DA REDE PURA (sem física, arquitetura v4.1 — inalterada)
# ===========================
function treinar_pura()
    params       = inicializar_params(1234)
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

# (A Rede Pura agora usa predict_pura — arquitetura antiga, sigmoide —
# de forma explícita e separada da predict() da PINN. Retreinar a Rede
# Pura dentro deste arquivo, se necessário algum dia, usará
# corretamente essa arquitetura antiga, sem risco de misturar com a
# mudança de positividade feita para a PINN.)

# ===========================
# 9. EXECUTAR OU CARREGAR
# ===========================
println("="^60)
println("  SIR-PINN — Treino: Malária Amazonas 2022-2024 (semanas 1-156)")
println("  β(t) modulado por clima (chuva lag=$(ATRASO_CHUVA), temp lag=$(ATRASO_TEMP), umid lag=$(ATRASO_UMID))")
println("  Projeção real: Malária Amazonas 2025 (semanas 157-209)")
println("  Física: SIR clássico, β(t) = exp(log_β₀ + c·clima(t))")
println("  Colocação: universo COMPLETO (210 pontos, sem amostragem)")
println("  Arquitetura [4→16→32→16→4]")
println("  NOVO v4.3: restrição HARD de positividade/monotonicidade em")
println("  dC/dt via softplus + integração cumulativa (sem penalidade)")
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
log_β0_ap  = params_pinn[IDX_log_β0]
c_chuva_ap = params_pinn[IDX_c_chuva]
c_temp_ap  = params_pinn[IDX_c_temp]
c_umid_ap  = params_pinn[IDX_c_umid]
γ_ap       = exp(params_pinn[IDX_log_γ])
β0_ap      = exp(log_β0_ap)

β_treino_vals = [beta_em(t, log_β0_ap, c_chuva_ap, c_temp_ap, c_umid_ap) for t in t_real]
β_min, β_max  = extrema(β_treino_vals)
R0_min, R0_max = β_min/γ_ap, β_max/γ_ap

println("\n" * "="^60)
println("  RESULTADO PINN — parâmetros estimados no treino (Malária 2022-2024)")
println("="^60)
@printf("  β₀ (nível base)     : %.4f\n", β0_ap)
@printf("  c_chuva              : %.4f\n", c_chuva_ap)
@printf("  c_temp                : %.4f\n", c_temp_ap)
@printf("  c_umid                : %.4f\n", c_umid_ap)
@printf("  γ estimado            : %.4f\n", γ_ap)
@printf("  β(t) no treino: mínimo %.4f — máximo %.4f\n", β_min, β_max)
@printf("  R₀(t) no treino: mínimo %.3f — máximo %.3f\n", R0_min, R0_max)
@printf("  Período infeccioso: %.1f semanas\n", 1.0/γ_ap)
println("="^62)

# ===========================
# 10b. CHECAGEM DA RESTRIÇÃO HARD  [NOVO]
# ===========================
# Como a positividade agora é garantida por construção (softplus nunca
# é negativo, para NENHUM valor real de entrada), esta checagem deve
# dar SEMPRE zero violações — em qualquer ponto, não só nos 210 pontos
# de colocação. Serve de teste de sanidade (confirma que não há bug na
# integração) e de comparação direta com a v4.2 (82 violações na
# projeção, mesmo com a penalidade soft ativa).
function checar_restricao_hard(params, label)
    t_fino = collect(0.0:0.1:T_HORIZON)
    taxas_finas = [predict(t, params)[4] for t in t_fino]
    n_negativas = count(<(0.0), taxas_finas)
    pior = minimum(taxas_finas)
    println("\n── Checagem de positividade (hard) — $label ──")
    @printf("  Pontos com taxa < 0 (deveria ser sempre 0): %d de %d\n", n_negativas, length(t_fino))
    @printf("  Pior taxa observada: %.8f (softplus nunca deveria produzir valor < 0)\n", pior)
end

checar_restricao_hard(params_pinn, "PINN (v4.3, restrição hard)")

# ===========================
# 11. PREDIÇÕES — TREINO (2022-2024) e PROJEÇÃO (2025)
# ===========================
t_plot_treino = collect(0.0:0.5:T_FINAL)
t_plot_teste  = collect(T_FINAL:0.5:(T_FINAL + Float64(N_TESTE)))

# --- PINN (arquitetura nova: taxa + integração) -------------------------
function predicoes_pinn_completas(params)
    # C em TODOS os pontos inteiros (0..209) de uma vez, já que
    # t_collocation cobre treino + projeção simultaneamente.
    C_grid_abs = C_grid_completo(params) .* C_ESCALA   # comprimento 210
    I_treino_pred = diff(C_grid_abs[1:N_TREINO+1])      # 156 valores (t=1..156)
    I_teste_pred  = diff(C_grid_abs[N_TREINO+1:end])    # 53 valores  (t=157..209)

    # Curvas finas (passo 0.5) para os gráficos:
    C_treino_fino = C_em_malha(t_plot_treino, params, 0.0) .* C_ESCALA
    C_fim_treino  = C_grid_abs[N_TREINO+1]  # já em escala absoluta
    C_teste_fino  = C_em_malha(t_plot_teste, params, C_fim_treino / C_ESCALA) .* C_ESCALA

    # S, I, R em malha fina, para o gráfico de compartimentos (p5)
    SIR_treino_fino = hcat([predict(t, params)[1:3] for t in t_plot_treino]...)

    return C_treino_fino, C_teste_fino, I_treino_pred, I_teste_pred, SIR_treino_fino
end

C_treino_pinn, C_teste_pinn, I_treino_pinn, I_teste_pinn, SIR_treino_pinn = predicoes_pinn_completas(params_pinn)

# --- Rede Pura (arquitetura antiga v4.1: sigmoide direto em C) ----------
# CORRIGIDO: usa predict_pura, não predict (ver seção 3c para o porquê)
function predicoes_pura(params)
    preds      = hcat([predict_pura(t, params) for t in t_plot_treino]...)
    C_pred_abs = preds[4,:] .* C_ESCALA
    C_at_t     = [predict_pura(t, params)[4] * C_ESCALA for t in t_real]
    C_prev     = vcat([0.0], C_at_t[1:end-1])
    I_pred     = C_at_t .- C_prev
    return C_pred_abs, I_pred
end

function predicoes_pura_teste(params)
    preds       = hcat([predict_pura(t, params) for t in t_plot_teste]...)
    C_pred_abs  = preds[4, :] .* C_ESCALA
    t_pred_full = vcat([T_FINAL], t_teste_abs)
    C_at_t_full = [predict_pura(t, params)[4] * C_ESCALA for t in t_pred_full]
    I_pred      = diff(C_at_t_full)
    return C_pred_abs, I_pred
end

C_treino_pura, I_treino_pura = predicoes_pura(params_pura)
C_teste_pura,  I_teste_pura  = predicoes_pura_teste(params_pura)

ts_rk4, S_rk4, I_rk4, R_rk4, C_rk4 = resolver_rk4(log_β0_ap, c_chuva_ap, c_temp_ap, c_umid_ap, γ_ap, T_FINAL)
C_rk4_abs = C_rk4 .* C_ESCALA

eqm(a, b)  = mean((a .- b).^2)
rmse(a, b) = sqrt(eqm(a, b))

# Para a cumulativa: PINN já vem calculada exatamente nos pontos ímpares
# de t_plot_treino/t_plot_teste (passo 0.5), então extraímos os pontos
# inteiros (índices 3,5,7,... já que t_plot_treino começa em 0.0 com
# passo 0.5 → índice 3 = t=1.0, igual à v4.1/v4.2 originais).
C_treino_pinn_pts = C_treino_pinn[3:2:end]
C_treino_pura_pts = C_treino_pura[3:2:end]
C_teste_pinn_pts  = C_teste_pinn[3:2:end]
C_teste_pura_pts  = C_teste_pura[3:2:end]

eqm_pinn_I_tr  = eqm(I_treino_pinn, I_treino_abs)
eqm_pura_I_tr  = eqm(I_treino_pura, I_treino_abs)
rmse_pinn_I_tr = rmse(I_treino_pinn, I_treino_abs)
rmse_pura_I_tr = rmse(I_treino_pura, I_treino_abs)
eqm_pinn_C_tr  = eqm(C_treino_pinn_pts, C_treino_abs)
eqm_pura_C_tr  = eqm(C_treino_pura_pts, C_treino_abs)
rmse_pinn_C_tr = rmse(C_treino_pinn_pts, C_treino_abs)
rmse_pura_C_tr = rmse(C_treino_pura_pts, C_treino_abs)

eqm_pinn_I_te  = eqm(I_teste_pinn, I_teste_abs)
eqm_pura_I_te  = eqm(I_teste_pura, I_teste_abs)
rmse_pinn_I_te = rmse(I_teste_pinn, I_teste_abs)
rmse_pura_I_te = rmse(I_teste_pura, I_teste_abs)
eqm_pinn_C_te  = eqm(C_teste_pinn_pts, C_teste_abs)
eqm_pura_C_te  = eqm(C_teste_pura_pts, C_teste_abs)
rmse_pinn_C_te = rmse(C_teste_pinn_pts, C_teste_abs)
rmse_pura_C_te = rmse(C_teste_pura_pts, C_teste_abs)

println("\n" * "="^62)
println("  MÉTRICAS — TREINO (Malária Amazonas 2022-2024) — Incidência Semanal")
println("="^62)
@printf("  %-30s %12s %12s\n", "Métrica", "PINN", "Rede Pura")
println("  " * "-"^58)
@printf("  %-30s %12.2f %12.2f\n", "EQM  (casos²)", eqm_pinn_I_tr,  eqm_pura_I_tr)
@printf("  %-30s %12.2f %12.2f\n", "RMSE (casos)",  rmse_pinn_I_tr, rmse_pura_I_tr)
println("="^62)

println("\n" * "="^62)
println("  MÉTRICAS — TREINO (Malária Amazonas 2022-2024) — Cumulativa")
println("="^62)
@printf("  %-30s %12s %12s\n", "Métrica", "PINN", "Rede Pura")
println("  " * "-"^58)
@printf("  %-30s %12.2f %12.2f\n", "EQM  (casos²)", eqm_pinn_C_tr,  eqm_pura_C_tr)
@printf("  %-30s %12.2f %12.2f\n", "RMSE (casos)",  rmse_pinn_C_tr, rmse_pura_C_tr)
println("="^62)

println("\n" * "="^62)
println("  MÉTRICAS — PROJEÇÃO (Malária Amazonas 2025) — Incidência Semanal")
println("  *** extrapolação real: nenhum dado de 2025 no treino ***")
println("="^62)
@printf("  %-30s %12s %12s\n", "Métrica", "PINN", "Rede Pura")
println("  " * "-"^58)
@printf("  %-30s %12.2f %12.2f\n", "EQM  (casos²)", eqm_pinn_I_te,  eqm_pura_I_te)
@printf("  %-30s %12.2f %12.2f\n", "RMSE (casos)",  rmse_pinn_I_te, rmse_pura_I_te)
println("="^62)

println("\n" * "="^62)
println("  MÉTRICAS — PROJEÇÃO (Malária Amazonas 2025) — Cumulativa")
println("  *** extrapolação real: nenhum dado de 2025 no treino ***")
println("="^62)
@printf("  %-30s %12s %12s\n", "Métrica", "PINN", "Rede Pura")
println("  " * "-"^58)
@printf("  %-30s %12.2f %12.2f\n", "EQM  (casos²)", eqm_pinn_C_te,  eqm_pura_C_te)
@printf("  %-30s %12.2f %12.2f\n", "RMSE (casos)",  rmse_pinn_C_te, rmse_pura_C_te)
println("="^62)

n_tr  = N_TREINO
k_mod = N_PARAMS

aic(eqm_val, n, k) = n * log(eqm_val) + 2*k
bic(eqm_val, n, k) = n * log(eqm_val) + k * log(n)

aic_pinn_tr = aic(eqm_pinn_I_tr, n_tr, k_mod)
aic_pura_tr = aic(eqm_pura_I_tr, n_tr, k_mod)
bic_pinn_tr = bic(eqm_pinn_I_tr, n_tr, k_mod)
bic_pura_tr = bic(eqm_pura_I_tr, n_tr, k_mod)

println("\n" * "="^62)
println("  TABELA AIC / BIC — Incidência Semanal (Treino Malária 2022-2024)")
println("  *** k >> n aqui (1225 >> 156) — aproximação assintótica do")
println("      AIC/BIC não é confiável nesse regime; use com cautela ***")
println("="^62)
@printf("  %-30s %12s %12s\n", "Critério", "PINN", "Rede Pura")
println("  " * "-"^58)
@printf("  %-30s %12d %12d\n", "n (pontos de dados)", n_tr, n_tr)
@printf("  %-30s %12d %12d\n", "k (parâmetros livres)", k_mod, k_mod)
@printf("  %-30s %12.2f %12.2f\n", "AIC", aic_pinn_tr, aic_pura_tr)
@printf("  %-30s %12.2f %12.2f\n", "BIC", bic_pinn_tr, bic_pura_tr)
println("="^62)

fator_pinn = eqm_pinn_I_te / eqm_pinn_I_tr
fator_pura = eqm_pura_I_te / eqm_pura_I_tr

println("\n" * "="^62)
println("  FATOR DE DEGRADAÇÃO (Projeção/Treino) — Incidência Semanal")
println("="^62)
@printf("  %-30s %12s %12s\n", "Modelo", "PINN", "Rede Pura")
println("  " * "-"^58)
@printf("  %-30s %11.2fx %11.2fx\n", "EQM projeção / EQM treino", fator_pinn, fator_pura)
println("="^62)

vies_pinn = mean(I_teste_pinn .- I_teste_abs)
vies_pura = mean(I_teste_pura .- I_teste_abs)
@printf("Viés médio (PINN): %.2f casos/semana\n", vies_pinn)
@printf("Viés médio (Pura): %.2f casos/semana\n", vies_pura)


# ===========================
# 12. GRÁFICOS
# ===========================
fs_tick=12; fs_guide=13; fs_title=14; fs_legend=10

p1 = plot(title="Treino — Incidência Semanal (Malária Amazonas 2022-2024)",
          xlabel="Semana epidemiológica (t)", ylabel="Casos novos",
          titlefontsize=fs_title, guidefontsize=fs_guide,
          tickfontsize=fs_tick, legendfontsize=fs_legend, left_margin=20mm,
          legend=:outerright)
scatter!(p1, t_real, I_treino_abs,
         label="Dados reais 2022-2024 — Malária Amazonas", markersize=3, color=:black)
plot!(p1, t_real, I_treino_pinn,
      label="PINN (RMSE=$(round(rmse_pinn_I_tr, digits=1)))",
      lw=2, color=:red)
plot!(p1, t_real, I_treino_pura,
      label="Rede pura (RMSE=$(round(rmse_pura_I_tr, digits=1)))",
      lw=2, color=:blue, ls=:dash)
vline!(p1, [52.0, 104.0, T_FINAL], label="", ls=:dot, color=:gray, lw=1)

p2 = plot(title="Projeção — Incidência Semanal (Malária Amazonas 2025) *** extrapolação real ***",
          xlabel="Semana epidemiológica (t, contínuo desde 2022)", ylabel="Casos novos",
          titlefontsize=fs_title, guidefontsize=fs_guide,
          tickfontsize=fs_tick, legendfontsize=fs_legend, left_margin=20mm,
          legend=:outerright)
scatter!(p2, t_teste_abs, I_teste_abs,
         label="Dados reais 2025 — Malária Amazonas", markersize=5, color=:black)
plot!(p2, t_teste_abs, I_teste_pinn,
      label="PINN (RMSE=$(round(rmse_pinn_I_te, digits=1)))",
      lw=2.5, color=:red)
plot!(p2, t_teste_abs, I_teste_pura,
      label="Rede pura (RMSE=$(round(rmse_pura_I_te, digits=1)))",
      lw=2.5, color=:blue, ls=:dash)
vline!(p2, [T_FINAL], label="", ls=:dot, color=:gray, lw=1)
hline!(p2, [0.0], label="", ls=:dot, color=:darkred, lw=1)

p3 = plot(title="Cumulativa C(t) — Treino 2022-2024 e Projeção 2025 (β₀=$(round(β0_ap,digits=3)) γ=$(round(γ_ap,digits=3)) R₀∈[$(round(R0_min,digits=2)),$(round(R0_max,digits=2))])",
          xlabel="Semana epidemiológica (t, contínuo)", ylabel="Casos acumulados",
          titlefontsize=fs_title, guidefontsize=fs_guide,
          tickfontsize=fs_tick, legendfontsize=fs_legend, left_margin=20mm,
          legend=:outerright)
scatter!(p3, t_real,      C_treino_abs, label="Real 2022-2024 — Malária Amazonas (treino)", markersize=3, color=:black)
scatter!(p3, t_teste_abs, C_teste_abs,  label="Real 2025 — Malária Amazonas (projeção)", markersize=4, color=:gray,  markershape=:diamond)
plot!(p3, t_plot_treino, C_treino_pinn, label="PINN treino", lw=2, color=:red)
plot!(p3, t_plot_teste,  C_teste_pinn,  label="PINN projeção",  lw=2, color=:red,  ls=:dash)
plot!(p3, t_plot_treino, C_treino_pura, label="Pura treino", lw=2, color=:blue)
plot!(p3, t_plot_teste,  C_teste_pura,  label="Pura projeção",  lw=2, color=:blue, ls=:dash)
vline!(p3, [T_FINAL], label="", ls=:dot, color=:gray, lw=1)

n_adam = EPOCHS_ADAM ÷ 500
n_lb   = max(0, min(length(hist_pinn), length(hist_pura)) - n_adam)
x_adam = collect(500:500:EPOCHS_ADAM)
x_lb   = collect((EPOCHS_ADAM+500):500:(EPOCHS_ADAM + n_lb*500))
x_hist = vcat(x_adam, x_lb)
n_plot = min(length(x_hist), length(hist_pinn), length(hist_pura))

p4 = plot(title="Histórico de Loss (log) — Adam + L-BFGS",
          xlabel="Iteração", ylabel="Loss", yscale=:log10,
          titlefontsize=fs_title, guidefontsize=fs_guide,
          tickfontsize=fs_tick, legendfontsize=fs_legend, left_margin=20mm,
          legend=:outerright)
plot!(p4, x_hist[1:n_plot], hist_pinn[1:n_plot],
      label="PINN (física + positividade hard)", lw=2, color=:red, marker=:circle, markersize=3)
plot!(p4, x_hist[1:n_plot], hist_pura[1:n_plot],
      label="Rede pura (sem física)", lw=2, color=:blue, marker=:circle, markersize=3)
vline!(p4, [EPOCHS_ADAM], label="Adam → L-BFGS", ls=:dash, color=:gray, lw=1.5)

# p5: agora mostra S,I,R (proporção) e, junto, dC/dt normalizado (taxa)
# — não mais C(t) diretamente, já que a 4ª saída da rede é a taxa.
p5 = plot(title="Compartimentos PINN vs RK4 (β₀=$(round(β0_ap,digits=3)) γ=$(round(γ_ap,digits=3)), β(t) climático)",
          xlabel="Semana epidemiológica", ylabel="Proporção da população / taxa",
          titlefontsize=fs_title, guidefontsize=fs_guide,
          tickfontsize=fs_tick, legendfontsize=fs_legend, left_margin=20mm,
          legend=:outerright)
plot!(p5, t_plot_treino, SIR_treino_pinn[1,:], lw=2.5, color=:blue,   label="PINN S(t)")
plot!(p5, t_plot_treino, SIR_treino_pinn[2,:], lw=2.5, color=:red,    label="PINN I(t)")
plot!(p5, t_plot_treino, SIR_treino_pinn[3,:], lw=2.5, color=:green,  label="PINN R(t)")
plot!(p5, ts_rk4, S_rk4, lw=1.5, ls=:dash, color=:darkblue,  label="RK4 S(t)", alpha=0.8)
plot!(p5, ts_rk4, I_rk4, lw=1.5, ls=:dash, color=:darkred,   label="RK4 I(t)", alpha=0.8)
plot!(p5, ts_rk4, R_rk4, lw=1.5, ls=:dash, color=:darkgreen, label="RK4 R(t)", alpha=0.8)
plot!(p5, ts_rk4, C_rk4, lw=1.5, ls=:dash, color=:magenta,   label="RK4 C(t)", alpha=0.8)

categorias_eqm = ["Treino\n(2022-2024)", "Projeção\n(2025)"]
eqm_pinn_comp  = [eqm_pinn_I_tr, eqm_pinn_I_te]
eqm_pura_comp  = [eqm_pura_I_tr, eqm_pura_I_te]

p6 = groupedbar(
    categorias_eqm,
    hcat(eqm_pinn_comp, eqm_pura_comp),
    label=["PINN (física + positividade hard)" "Rede pura (sem física)"],
    color=[:red :blue],
    alpha=0.75,
    title="EQM — Treino vs Projeção (Incidência Semanal)",
    ylabel="EQM (casos²)",
    titlefontsize=fs_title, guidefontsize=fs_guide,
    tickfontsize=fs_tick, legendfontsize=fs_legend,
    left_margin=20mm, bar_width=0.6,
    legend=:outerright
)

categorias_fator = ["PINN", "Rede Pura"]
fatores_degradacao = [fator_pinn, fator_pura]

p7 = bar(
    categorias_fator,
    fatores_degradacao,
    label=false,
    color=[:red, :blue],
    alpha=0.75,
    title="Fator de Degradação (EQM Projeção / EQM Treino) — Incidência Semanal",
    ylabel="Fator (×)",
    titlefontsize=fs_title, guidefontsize=fs_guide,
    tickfontsize=fs_tick, legendfontsize=fs_legend,
    left_margin=20mm, bar_width=0.5
)
annotate!(p7, 1, fatores_degradacao[1] + 0.15*maximum(fatores_degradacao),
          text("$(round(fatores_degradacao[1], digits=2))×", fs_guide, :black))
annotate!(p7, 2, fatores_degradacao[2] + 0.15*maximum(fatores_degradacao),
          text("$(round(fatores_degradacao[2], digits=2))×", fs_guide, :black))

pfinal = plot(p1, p2, p3, p4, p5, p6, p7, layout=(7,1), size=(1300,2800), dpi=300)
savefig(pfinal, joinpath(@__DIR__, "sir_pinn_projecao_v4_3_positividade_hard.png"))
display(pfinal)
println("\nGráfico salvo em sir_pinn_projecao_v4_3_positividade_hard.png")