# Guia de uso do booster_train (T1)

Este guia cobre o fluxo completo de treino no `booster_train` até o deploy no `booster_deploy`, e explica como tirar a cabeça da saída da rede (21 juntas em vez de 23), já que no nosso robô quem move a cabeça é o brain.

> **Estado deste documento.** Tudo aqui foi levantado lendo o código dos dois repositórios. A receita de 21 juntas (seção 5) **ainda não foi executada**: a máquina onde o guia foi escrito não tem Isaac Lab. Os pontos que dependem de comportamento do Isaac Lab e não foram conferidos estão marcados com "a confirmar".

## 1. O que o repositório treina

O `booster_train` roda sobre Isaac Lab (testado pela Booster com Isaac Lab 2.3.2 e Isaac Sim 5.1.0) e usa PPO do RSL-RL. Hoje ele só tem tarefas de **imitação de movimento** (BeyondMimic): a política aprende a reproduzir um movimento de referência gravado em arquivo. Serve para chute, levantar, dança. Não há tarefa de caminhada por comando de velocidade.

Tarefas registradas:

| Tarefa | Robô | Config de ambiente |
|---|---|---|
| `Booster-T1-MotionTracking-v0` | T1 | `RoughWoStateEstimationEnvCfg` |
| `Booster-T1-MotionTracking-v0-Play` | T1 | `PlayFlatWoStateEstimationEnvCfg` |
| `Booster-T2-MotionTracking-v0` e `-Rough-v0` (com `-Play`) | T2 | |
| `Booster-K1-MJ_Dance_002-v0` (com `-Play`) | K1 | |

Arquivos que importam para o T1, a partir de `source/booster_train/booster_train/`:

| Arquivo | Conteúdo |
|---|---|
| `tasks/manager_based/beyond_mimic/robots/t1/motion_tracking/tracking_env_cfg.py` | Cena, ações, observações, recompensas, eventos e terminações |
| `.../t1/motion_tracking/env_cfg.py` | Variações da tarefa: robô, movimento de referência, terreno |
| `.../t1/motion_tracking/ppo_cfg.py` | Nome do experimento e número de iterações |
| `.../t1/motion_tracking/__init__.py` | Registro das tarefas (`gym.register`) |
| `tasks/manager_based/beyond_mimic/agents/rsl_rl_ppo_cfg.py` | Hiperparâmetros do PPO e tamanho da rede |
| `tasks/manager_based/beyond_mimic/mdp/` | Funções de comando, observação, recompensa e terminação |
| `assets/robots/booster_t1.py` | Modelo do robô, atuadores e `T1_ACTION_SCALE` |

## 2. Fluxo de trabalho

Todos os comandos rodam na raiz do `booster_train`, com o Python que tem o Isaac Lab instalado.

### 2.1 Instalação

```bash
python -m pip install -e source/booster_train
```

Também é preciso clonar e instalar o `booster_assets` (modelos dos robôs e movimentos de exemplo).

### 2.2 Preparar o movimento de referência

```bash
python scripts/csv_to_npz.py --headless --robot t1 \
    --input_file=<MOVIMENTO>.csv --input_fps=<FPS> \
    --output_file=<MOVIMENTO>.npz
```

- **`--robot` tem padrão `k1`.** Para o T1 é obrigatório passar `--robot t1`.
- `--output_fps` tem padrão 50, que é a frequência da política. Não mude.
- `--frame_range INICIO FIM` recorta o movimento (índices a partir de 1, inclusivos).
- A ordem das juntas esperada no CSV está em `assets/robots/motion_joint_names.py` (`T1_CSV_JOINT_NAMES`): cabeça, braço esquerdo, braço direito, cintura, perna esquerda, perna direita.

O movimento usado pela tarefa é definido em `env_cfg.py`:

```python
self.commands.motion.motion_file = "t1_dance.npz"
```

O caminho é relativo ao diretório de onde o treino é lançado, e o carregador falha com `Invalid file path` se o arquivo não estiver lá. Use caminho absoluto para evitar surpresa.

### 2.3 Treinar

```bash
python scripts/list_envs.py                      # lista as tarefas
python scripts/rsl_rl/train.py --task=Booster-T1-MotionTracking-v0 --headless --device cuda:0
```

Opções úteis: `--num_envs`, `--max_iterations`, `--seed`, `--run_name <sufixo>`, e para continuar um treino `--resume --load_run <pasta> --checkpoint <arquivo>`.

Os logs ficam em `logs/rsl_rl/<experiment_name>/<data-hora>[_<run_name>]/`, com um checkpoint `model_<N>.pt` a cada 1000 iterações.

Valores padrão do T1:

| Parâmetro | Valor | Onde |
|---|---|---|
| Ambientes em paralelo | 4096 | `tracking_env_cfg.py` |
| Passo da física | 5 ms | `tracking_env_cfg.py` |
| Decimação | 4 (política a 50 Hz) | `tracking_env_cfg.py` |
| Duração do episódio | 10 s | `tracking_env_cfg.py` |
| Iterações | 50000 | `ppo_cfg.py` |
| Rede (ator e crítico) | 512, 256, 128 com ELU | `rsl_rl_ppo_cfg.py` |
| Normalização de observações | ligada | `rsl_rl_ppo_cfg.py` |
| `experiment_name` | `t1_motion_tracking` | `ppo_cfg.py` |

### 2.4 Visualizar e exportar

```bash
python scripts/rsl_rl/play.py --task=Booster-T1-MotionTracking-v0-Play --checkpoint=<CAMINHO>/model_<N>.pt
```

Além de mostrar a política, o `play.py` exporta o modelo para `logs/rsl_rl/<experimento>/<execução>/exported/`, com o nome `<experiment_name>_<execução>.pt` (TorchScript) e `.onnx`. Com `--headless` ele só exporta e sai.

O arquivo exportado já contém o normalizador de observações. No deploy, a observação entra crua.

## 3. Entradas e saídas da rede (T1 de fábrica)

A tarefa padrão do T1 é a variante sem estimação de estado: ela remove `motion_anchor_pos_b` e `base_lin_vel` do ator, porque o robô real não mede essas grandezas com confiança.

| Termo (nesta ordem) | Dimensão |
|---|---|
| `command`: posição e velocidade de referência das juntas | 46 |
| `motion_anchor_ori_b`: orientação do tronco de referência em relação ao tronco atual | 6 |
| `base_ang_vel` | 3 |
| `joint_pos` (relativa à pose padrão) | 23 |
| `joint_vel` | 23 |
| `actions` (última ação) | 23 |
| **Total** | **124** |

Saída: 23 valores, um por junta, **na ordem do Isaac Lab**, que não é a ordem do robô real:

```
aahead_yaw, left_shoulder_pitch, right_shoulder_pitch, waist_yaw, aahead_pitch,
left_shoulder_roll, right_shoulder_roll, left_hip_pitch, right_hip_pitch,
left_elbow_pitch, right_elbow_pitch, left_hip_roll, right_hip_roll,
left_elbow_yaw, right_elbow_yaw, left_hip_yaw, right_hip_yaw,
left_knee_pitch, right_knee_pitch, left_ankle_pitch, right_ankle_pitch,
left_ankle_roll, right_ankle_roll
```

A ação vira alvo de posição assim: `alvo = pose_padrão + ação × escala`. A escala por junta é `coeficiente × limite_de_torque / Kp`, calculada em `booster_t1.py` (`T1_ACTION_SCALE`). O coeficiente é 0.25 quando a junta não aparece em `per_joint_scale_coefficient`, que é o caso da cabeça.

O `NeuralNets/model_best.pt` tem exatamente esse formato (124 entradas, 23 saídas).

## 4. Como o tamanho da saída é definido

Não existe um número "23" na configuração da rede. O RSL-RL cria a última camada com o tamanho do espaço de ações do ambiente, e esse espaço vem do termo de ação em `tracking_env_cfg.py`:

```python
joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)
```

`joint_names` aceita expressões regulares. `".*"` casa com todas as juntas, por isso 23. Restringir a lista muda a saída da rede sem tocar na arquitetura.

## 5. Tirar a cabeça da saída (23 → 21)

A ideia é criar uma variante da tarefa, sem alterar a original, para poder comparar as duas.

### 5.1 Nova config de ambiente

Em `.../t1/motion_tracking/env_cfg.py`:

```python
@configclass
class NoHeadEnvCfg(RoughWoStateEstimationEnvCfg):
    """A política controla 21 juntas; a cabeça fica com o brain."""

    def __post_init__(self):
        super().__post_init__()
        # 1. A ação cobre todas as juntas, menos as da cabeça.
        self.actions.joint_pos.joint_names = ["^(?!.*head).*$"]
        # 2. A escala não pode ter entrada para juntas fora da ação.
        self.actions.joint_pos.scale = {
            k: v for k, v in T1_ACTION_SCALE.items() if "head" not in k
        }
        # 3. A cabeça sai dos corpos rastreados pela recompensa.
        self.commands.motion.body_names = [
            b for b in self.commands.motion.body_names if "head" not in b
        ]


@configclass
class PlayNoHeadEnvCfg(NoHeadEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.commands.motion.play = True
        self.events.push_robot = None
```

Por que cada item:

1. **Ação.** As juntas da cabeça no T1 são `aahead_yaw_joint` e `aahead_pitch_joint`. A expressão casa com qualquer nome que não contenha `head`.
2. **Escala.** As chaves de `T1_ACTION_SCALE` são os padrões dos atuadores, e a da cabeça é `".*head.*"`. O Isaac Lab recusa um dicionário de escala com chave que não casa com nenhuma junta da ação (a confirmar na primeira execução; se não reclamar, o filtro é inofensivo).
3. **Recompensa.** `aahead_pitch_link` está na lista de corpos rastreados. Se ficar, a política é penalizada pelo erro de um corpo que ela não controla.

### 5.2 Registrar a tarefa

Em `.../t1/motion_tracking/__init__.py`:

```python
gym.register(
    id="Booster-T1-MotionTracking-NoHead-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:NoHeadEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-T1-MotionTracking-NoHead-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PlayNoHeadEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)
```

Vale criar também um `PPORunnerCfg` próprio com outro `experiment_name` (por exemplo `t1_motion_tracking_nohead`), para os logs não se misturarem com os do modelo de 23 juntas.

### 5.3 O que muda na rede

| | De fábrica | Sem cabeça |
|---|---|---|
| Saída | 23 | 21 |
| `actions` na observação | 23 | 21 |
| Entrada total | 124 | 122 |

`joint_pos`, `joint_vel` e `command` continuam com 23 juntas. Isso é intencional: a política enxerga onde a cabeça está e pode compensar o movimento dela, só não a comanda.

A ordem das 21 saídas é a ordem do Isaac Lab da seção 3 sem os índices 0 (`aahead_yaw`) e 4 (`aahead_pitch`).

Um checkpoint de 23 saídas **não pode** ser reaproveitado com `--resume` nessa tarefa: as dimensões da primeira e da última camada mudam. O treino começa do zero.

### 5.4 Movimentar a cabeça durante o treino (recomendado)

Sem termo de ação, ninguém comanda a cabeça na simulação: a cada reinício ela é posta na pose do movimento de referência e depois fica presa pelo PD do atuador. No robô, o brain vai movê-la o tempo todo para procurar a bola. Para a política treinar com essa perturbação, adicione um evento que sorteie alvos para a cabeça.

Esboço, a colocar em `mdp/events.py` (o arquivo já importa `torch` e `math_utils`):

```python
def randomize_head_target(env, env_ids, asset_cfg, yaw_range, pitch_range):
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    yaw = math_utils.sample_uniform(*yaw_range, (n, 1), asset.device)
    pitch = math_utils.sample_uniform(*pitch_range, (n, 1), asset.device)
    asset.set_joint_position_target(
        torch.cat([yaw, pitch], dim=1), joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )
```

E na config:

```python
self.events.head_motion = EventTerm(
    func=mdp.randomize_head_target,
    mode="interval",
    interval_range_s=(0.5, 2.0),
    params={
        "asset_cfg": SceneEntityCfg(
            "robot", joint_names=["aahead_yaw_joint", "aahead_pitch_joint"], preserve_order=True
        ),
        "yaw_range": (-1.0, 1.0),
        "pitch_range": (-0.3, 0.8),
    },
)
```

Pontos a confirmar ao implementar:

- As faixas acima são um chute. Use as que o brain realmente comanda, dentro dos limites da junta no URDF.
- Estou supondo que o alvo definido pelo evento persiste entre passos, já que o termo de ação só escreve nas 21 juntas dele.
- O alvo muda em degrau. Se o brain move a cabeça de forma suave, vale interpolar.

A cabeça do T1 é leve, então o efeito no equilíbrio tende a ser pequeno. O evento é barato e cobre os movimentos rápidos de busca.

## 6. Deploy no booster_deploy

A tarefa de deploy correspondente é `tasks/beyond_mimic/` (`t1_motion_tracking`). O backend de inferência é escolhido pela extensão do arquivo: `.pt` usa TorchScript, `.onnx` usa ONNX Runtime.

Para o modelo de 23 juntas basta copiar o arquivo exportado e o `.npz` do movimento para `tasks/beyond_mimic/robots/t1/models/` e `.../motions/`, e apontar `checkpoint_path` e `motion_path`.

Para o modelo de 21 juntas, `tasks/beyond_mimic/beyond_mimic.py` precisa de três ajustes, porque hoje ele assume uma ação por junta:

1. **`last_action`** é criada com `robot.num_joints` (23) em `reset()`. Precisa ter 21.
2. **Saída.** O retorno faz `action[sim2real_map] * self.action_scale + self.default_joint_pos`, com 23 posições. A ação de 21 precisa ser expandida antes:

   ```python
   # em __init__: índices, na ordem do Isaac Lab, das juntas que a política controla
   sim_names = self.robot.cfg.sim_joint_names
   self.policy_joint_ids = [i for i, n in enumerate(sim_names) if "head" not in n]

   # em inference(), antes do retorno
   full = torch.zeros(self.robot.num_joints, dtype=torch.float32, device=self.cfg.device)
   full[self.policy_joint_ids] = action
   return full[sim2real_map] * self.action_scale + self.default_joint_pos
   ```

   Com ação zero, o alvo da cabeça fica na pose padrão.
3. **Observação.** `joint_pos` e `joint_vel` continuam com 23, só `last_action` cai para 21. Total 122, na mesma ordem da seção 5.3.

O ideal é deixar isso atrás de uma opção na `BeyondMimicPolicyCfg` (por exemplo uma lista de juntas excluídas), para a mesma classe servir aos dois modelos.

**Quem manda na cabeça.** O `policy_manager` já sobrescreve o alvo das duas juntas da cabeça com o que chega em `/policy_manager/head` (`policy_switcher.py`, `head_target`), depois de a política calcular a saída. Ou seja, rodando pelo `policy_manager`, a cabeça já é do brain mesmo com um modelo de 23 saídas. O ganho do modelo de 21 é a política treinar sabendo disso, em vez de aprender a usar uma junta que no robô não obedece a ela. Rodando direto por `scripts/deploy.py`, sem `policy_manager`, não há essa sobrescrita.

**Escala de ação.** `action_scale_factors` em `tasks/beyond_mimic/robots/t1/__init__.py` repete, na ordem do robô real, os coeficientes de `per_joint_scale_coefficient` do treino. Se mudar um lado, mude o outro.

## 7. Cuidados de sim2real

- **Frequência.** A política foi treinada a 50 Hz. O deploy tem que rodar em 50 Hz.
- **Ganhos PD e escala.** A escala de ação depende de `Kp` e do limite de torque. Mudou parâmetro de atuador, recalcule a escala e atualize o deploy.
- **Ordem das juntas.** Treino usa a ordem do Isaac Lab; robô usa a ordem serial (cabeça, braços, cintura, pernas). O `booster_deploy` converte com `sim2real_joint_indexes` e `real2sim_joint_indexes`. Erro aqui faz o robô se debater logo no primeiro passo; teste sempre em MuJoCo antes.
- **Atraso de comando.** Os atuadores do T1 sorteiam de 2 a 8 passos de física de atraso (10 a 40 ms). Se o passo da física mudar, a faixa em milissegundos muda junto.
- **Juntas paralelas do tornozelo.** O `Kd` da simulação inclui uma razão de inércia que não vale para o motor real. As notas do `README.md` (escritas para o T2) explicam a conversão; não copie o `Kd` da simulação para o robô.
- **Estimação de estado.** Use as variantes `WoStateEstimation`. As que observam velocidade linear da base e posição da âncora dependem de grandezas que o robô real não mede bem.

## 8. Problemas comuns

| Sintoma | Causa provável |
|---|---|
| `Invalid file path: t1_dance.npz` | Movimento não está no diretório de execução; use caminho absoluto em `motion_file` |
| Movimento convertido fica torto ou com juntas trocadas | `csv_to_npz.py` rodou sem `--robot t1` |
| Erro de dimensão ao dar `--resume` | Checkpoint de outra variante (23 contra 21 saídas, ou com e sem estimação de estado) |
| Robô se debate no primeiro passo do deploy | Ordem das juntas ou escala de ação diferente da do treino |
| Muito texto impresso ao importar a tarefa | `booster_t1.py` imprime a config dos atuadores e a escala na importação; é esperado |
