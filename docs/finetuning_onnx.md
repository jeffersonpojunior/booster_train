# Fine-tuning a partir de um modelo `.onnx`

## Resposta curta

**Não diretamente.** O `train.py` só retoma treino a partir de um checkpoint do RSL-RL (`model_<N>.pt`). O `.onnx` é um formato de deploy e não guarda tudo o que o treino precisa.

Há dois caminhos:

| Você tem | O que fazer | Seção |
|---|---|---|
| O checkpoint `model_<N>.pt` do treino original | Fine-tuning normal com `--resume`. É o caminho recomendado. | 2 |
| Só o `.onnx`, exportado por este repositório (ou com rede idêntica) | Converter o `.onnx` em checkpoint com `scripts/rsl_rl/onnx_to_checkpoint.py` e retomar. O ator é aproveitado, o crítico recomeça do zero. | 3 |
| Um `.onnx` de outro pipeline, com outra entrada, saída ou camadas | Não dá para aproveitar aqui. | 4 |

> **Estado deste documento.** O conversor da seção 3 foi testado fora do simulador, contra o `rsl-rl-lib` 3.1.2 e o exportador do Isaac Lab 2.3.2 (as versões que este repositório usa): o checkpoint gerado carrega sem erro e reproduz a saída do `.onnx` com diferença de 1e-7. O fluxo completo dentro do Isaac Lab (gerar o molde, retomar e treinar) **não foi executado**, porque a máquina onde o guia foi escrito não tem Isaac Lab.

## 1. Por que o `.onnx` não basta

O `play.py` exporta só o que o robô precisa para rodar a política. Comparando com o checkpoint de treino:

| Conteúdo | `model_<N>.pt` | `.onnx` exportado |
|---|---|---|
| Ator (MLP 512, 256, 128) | sim | sim |
| Normalizador de observações do ator | média, variância e contagem | média e desvio, embutidos no grafo |
| Crítico e seu normalizador | sim | **não** |
| Ruído de exploração (`std` das ações) | sim | **não** |
| Estado do otimizador (Adam) | sim | **não** |
| Número da iteração | sim | **não** |

O `exported/<nome>.pt` que sai junto com o `.onnx` é TorchScript e tem a mesma limitação. Ele também não serve para `--resume`.

O PPO precisa do crítico para calcular a vantagem de cada ação. Sem ele, o treino até roda, mas as primeiras atualizações do ator são guiadas por um crítico aleatório. Esse é o custo do caminho da seção 3.

## 2. Com o checkpoint `.pt` (recomendado)

Os checkpoints ficam em `logs/rsl_rl/<experiment_name>/<execução>/model_<N>.pt`. Para continuar a partir de um deles:

```bash
python scripts/rsl_rl/train.py --task=Booster-T1-MotionTracking-v0 --headless --device cuda:0 \
    --resume --load_run <PASTA_DA_EXECUÇÃO> --checkpoint model_<N>.pt \
    --max_iterations 5000 --run_name finetune
```

- `--load_run` e `--checkpoint` são nomes (ou expressões regulares) procurados dentro de `logs/rsl_rl/<experiment_name>/`, relativo ao diretório de onde o comando é lançado. Se o `.pt` veio de outra máquina, copie-o para uma pasta nesse caminho.
- O resultado vai para uma pasta nova (`<data-hora>_finetune`). A execução original não é alterada.
- `--max_iterations` é o número de iterações **a mais**. A contagem continua de `N`.
- O `--resume` restaura ator, crítico, normalizadores, ruído e otimizador.

O que pode mudar entre o treino original e o fine-tuning:

| Mudança | Funciona? |
|---|---|
| Movimento de referência (`motion_file` em `env_cfg.py`) | Sim |
| Recompensas, randomizações, terreno, terminações | Sim |
| Hiperparâmetros do PPO (taxa de aprendizado, entropia) | Sim |
| Termos ou dimensões das observações do ator ou do crítico | Não, erro de dimensão ao carregar |
| Número de juntas comandadas (23 para 21, por exemplo) | Não |
| Camadas ocultas ou função de ativação | Não |

Hiperparâmetros podem ser trocados na linha de comando, sem editar arquivo:

```bash
... --resume ... agent.algorithm.learning_rate=1e-4 agent.algorithm.entropy_coef=0.001
```

Com `schedule="adaptive"` (o padrão daqui), a taxa de aprendizado é reajustada a cada atualização pelo KL, então o valor passado vale só como ponto de partida.

## 3. Só com o `.onnx`

O script `scripts/rsl_rl/onnx_to_checkpoint.py` monta um checkpoint válido copiando o ator e o normalizador do `.onnx` para dentro de um checkpoint "molde" da tarefa de destino. O molde fornece o que falta (crítico, otimizador) no formato exato que a versão instalada do RSL-RL espera.

### 3.1 Requisitos

O `.onnx` precisa ter exatamente a mesma rede da tarefa de destino:

- mesmo número de observações na entrada e de ações na saída;
- mesmas camadas ocultas (aqui, 512, 256, 128) e mesma ativação (ELU);
- política sem memória (MLP). LSTM e GRU não são aceitos.

Para o T1 de fábrica isso significa entrada 124 e saída 23. O script confere as dimensões e aborta com as duas listas lado a lado se não baterem. A ativação ele só informa, não tem como conferir contra o molde.

A **ordem** das observações e das ações também precisa ser a mesma, e isso nenhum script consegue verificar: as dimensões podem bater com o conteúdo trocado. Só use um `.onnx` cuja origem você conhece.

### 3.2 Passo a passo

**1. Gerar o molde.** Uma iteração de treino da tarefa de destino, com poucos ambientes:

```bash
python scripts/rsl_rl/train.py --task=Booster-T1-MotionTracking-v0 --headless \
    --num_envs 64 --max_iterations 1 --run_name molde
```

Isso cria `logs/rsl_rl/t1_motion_tracking/<data-hora>_molde/model_0.pt`.

**2. Converter.** Não precisa do simulador, só de `torch`, `onnx` e `numpy`:

```bash
python scripts/rsl_rl/onnx_to_checkpoint.py \
    --onnx <POLITICA>.onnx \
    --template logs/rsl_rl/t1_motion_tracking/<data-hora>_molde/model_0.pt \
    --output logs/rsl_rl/t1_motion_tracking/from_onnx/model_0.pt \
    --action_std 0.3
```

| Opção | Efeito |
|---|---|
| `--action_std` | Ruído de exploração inicial. Sem a opção fica o do molde, 1.0, que é o valor de início de treino e desfaz rápido o que o ator já sabe. |
| `--normalizer_count` | Quantas amostras o normalizador "acha" que já viu (padrão 1e9). Quanto maior, mais devagar as estatísticas vindas do `.onnx` se movem durante o fine-tuning. |

O valor 0.3 para `--action_std` é um ponto de partida, não foi validado em treino.

**3. Retomar o treino** a partir do checkpoint convertido:

```bash
python scripts/rsl_rl/train.py --task=Booster-T1-MotionTracking-v0 --headless --device cuda:0 \
    --resume --load_run from_onnx --checkpoint model_0.pt \
    --run_name finetune_onnx agent.algorithm.learning_rate=1e-4
```

**4. Conferir antes de gastar horas de GPU.** Rode o `play.py` no checkpoint convertido. O robô deve se comportar como o `.onnx` original:

```bash
python scripts/rsl_rl/play.py --task=Booster-T1-MotionTracking-v0-Play \
    --checkpoint logs/rsl_rl/t1_motion_tracking/from_onnx/model_0.pt
```

### 3.3 O que esperar

- **O crítico começa aleatório.** Nas primeiras centenas de iterações a recompensa pode cair antes de se recuperar. Ruído inicial baixo e taxa de aprendizado baixa reduzem o estrago, não eliminam.
- **Não há como congelar o ator** enquanto o crítico aprende. O repositório não tem essa opção e ela exigiria mudar o código de treino.
- **O ruído de exploração original se perde.** O valor passado em `--action_std` é um chute.
- **Compare com um treino do zero.** Se a curva do fine-tuning cair até o nível de um treino novo e subir no mesmo ritmo, o `.onnx` não ajudou.

Se o checkpoint `.pt` original existir em algum lugar, vale mais a pena recuperá-lo do que seguir por aqui.

## 4. Modelos de outros pipelines

Um `.onnx` treinado fora deste repositório só serve se cumprir os requisitos da seção 3.1. Os modelos em `treinamento/NeuralNets/` não cumprem:

| Modelo | Entrada | Saída | Camadas ocultas |
|---|---|---|---|
| Tarefa T1 deste repositório | 124 | 23 | 512, 256, 128 |
| `Walk/T1_walk.onnx` | 365 | 14 | 256, 128, 128 |
| `Walk/T1_kick_inner.onnx` | 62 | 14 | 256, 128, 128 |
| `GetUpEngine/T1.onnx` | 76 | 23 | 256, 128, 128 |
| `GetUpEngine/T1_fast_standup.onnx` | 75 | 23 | 512, 256, 128 |

Eles vêm de tarefas com outras observações e, no caso dos de `Walk/`, outro conjunto de juntas. Nenhum deles tem normalizador embutido. Para ajustá-los é preciso o código de treino e o checkpoint do pipeline que os gerou. O `booster_train` hoje só tem tarefas de imitação de movimento.

Para inspecionar qualquer `.onnx`, abra-o em https://netron.app ou rode:

```bash
python -c "
import onnx, sys
g = onnx.load(sys.argv[1]).graph
print('entrada:', [d.dim_value for d in g.input[0].type.tensor_type.shape.dim])
print('saida:  ', [d.dim_value for d in g.output[0].type.tensor_type.shape.dim])
print('pesos:  ', [(i.name, list(i.dims)) for i in g.initializer])
" <MODELO>.onnx
```

## 5. Problemas comuns

| Sintoma | Causa provável |
|---|---|
| `size mismatch` ou `Missing key(s)` ao dar `--resume` | Checkpoint de outra variante da tarefa (observações, juntas ou camadas diferentes) |
| `The ONNX actor does not match the actor of the template checkpoint` | O `.onnx` não é da mesma tarefa do molde. Compare as duas listas impressas. |
| `--resume` não acha o checkpoint | A pasta não está em `logs/rsl_rl/<experiment_name>/`, ou o comando foi lançado de outro diretório |
| Tentou `--resume` com o `.pt` de `exported/` | Esse arquivo é TorchScript, não checkpoint. Use o `model_<N>.pt` ou converta o `.onnx`. |
| Recompensa despenca logo após retomar do `.onnx` | Crítico aleatório com ruído alto. Reduza `--action_std` e `agent.algorithm.learning_rate`. |
| `play.py` no checkpoint convertido se comporta diferente do `.onnx` | Ordem das observações ou das ações diferente entre a origem do `.onnx` e a tarefa, ou ativação diferente de ELU |
