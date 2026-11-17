# Do vídeo ao CSV: movimento de referência para o T1

Este documento mostra como transformar um vídeo (MP4) de uma pessoa fazendo um movimento no CSV que o `scripts/csv_to_npz.py` aceita. A partir do `.npz`, o treino segue o [guia de uso](guia_de_uso.md), seção 2.

O `booster_train` não lê vídeo. O caminho usa duas ferramentas externas:

```
vídeo MP4 ──GVHMR──▶ pose humana 3D (SMPL-X) ──GMR──▶ movimento do T1 (.pkl) ──▶ CSV ──csv_to_npz.py──▶ .npz
```

| Etapa | Ferramenta | Saída |
|---|---|---|
| 1. Estimar a pose humana 3D no mundo | GVHMR | `hmr4d_results.pt` |
| 2. Retargeting do humano para o T1 | GMR (`gvhmr_to_robot.py`) | `.pkl` com raiz e juntas do T1 |
| 3. Converter para CSV | GMR (`batch_gmr_pkl_to_csv.py`) | `.csv` sem cabeçalho |
| 4. Limpar (opcional, recomendado) | script do apêndice A | `.csv` recortado, suavizado, com o chão corrigido |
| 5. Gerar o `.npz` e conferir | `csv_to_npz.py` e `replay_npz.py` | `.npz` pronto para o treino |

> **Estado deste documento.** O formato do CSV e a ordem das juntas foram conferidos no código do `booster_train` e do GMR. O script do apêndice A foi testado sobre o `t1_dance.csv`. O pipeline completo, de um vídeo até o treino, **ainda não foi executado** aqui. Os pontos não conferidos estão marcados com "a confirmar". Os comandos do GVHMR e do GMR seguem os READMEs desses projetos; se algo mudar, eles valem mais que este documento.

## 1. O formato que o `csv_to_npz.py` espera

Conferido em `scripts/csv_to_npz.py` e no README do `booster_assets`:

| Colunas | Conteúdo | Unidade |
|---|---|---|
| 0–2 | Posição da raiz (`trunk`) no mundo: `x, y, z`, com z para cima | m |
| 3–6 | Orientação da raiz como quaternion **`x, y, z, w`** (o script converte para `wxyz`) | |
| 7–29 | As 23 juntas na ordem de `T1_CSV_JOINT_NAMES` | rad |

- Uma linha por frame, separada por vírgula. Cabeçalho é opcional; se existir, os nomes das juntas são conferidos contra a ordem esperada.
- Total de 30 colunas. Com outro número, o script para com erro.
- O FPS não fica no arquivo. Ele entra pelo `--input_fps`.

Ordem das juntas (`source/booster_train/booster_train/assets/robots/motion_joint_names.py`):

```
aahead_yaw, aahead_pitch,
left_shoulder_pitch, left_shoulder_roll, left_elbow_pitch, left_elbow_yaw,
right_shoulder_pitch, right_shoulder_roll, right_elbow_pitch, right_elbow_yaw,
waist_yaw,
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee_pitch, left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee_pitch, right_ankle_pitch, right_ankle_roll
```

O `booster_assets/motions/T1/t1_dance.csv` é um exemplo válido do formato.

## 2. Gravar o vídeo

A qualidade da reconstrução depende muito do vídeo. Antes de gravar:

- **Câmera fixa**, num tripé. Com câmera parada, o GVHMR roda no modo `-s` e a trajetória no mundo fica bem melhor.
- **Corpo inteiro no quadro o tempo todo**, pés incluídos. Pé cortado vira pé deslizando no robô.
- **Uma pessoa só** em cena, com roupa justa e fundo que contraste com ela.
- **Boa iluminação e pouco borrão de movimento.** Para movimentos rápidos, como um chute, grave a 60 fps ou mais.
- Comece e termine com a pessoa **parada em pé**, alguns segundos. Isso facilita recortar e dá ao robô um início estável.
- O movimento tem que caber no T1. Ele é bem menor que um humano e tem 4 juntas por braço, sem punho, e 1 junta de cintura (só yaw). Giros de tronco para frente ou para o lado, pulos altos e mãos complexas não vão ser reproduzidos.

## 3. Etapa 1: vídeo → pose humana 3D (GVHMR)

O GVHMR (`github.com/zju3dv/GVHMR`) estima o corpo SMPL-X em cada frame **e** a trajetória da pessoa no mundo. A trajetória é o que dá a posição da raiz. Estimadores que só dão a pose em relação à câmera, como o 4D-Humans/HMR2, não servem aqui.

Instalação: siga o `docs/INSTALL.md` do GVHMR. Ele pede um ambiente próprio, os pesos pré-treinados e os modelos de corpo SMPL e SMPL-X. Os modelos de corpo exigem cadastro nos sites oficiais (`smpl.is.tue.mpg.de` e `smpl-x.is.tue.mpg.de`) e têm licença só para pesquisa.

Execução:

```bash
cd <GVHMR>
python tools/demo/demo.py --video=<CAMINHO>/movimento.mp4 -s
```

- `-s` diz que a câmera está parada. Sem ele, o GVHMR estima o movimento da câmera também, o que é mais lento e erra mais.
- Saída: `outputs/demo/movimento/hmr4d_results.pt`, mais vídeos de visualização na mesma pasta. **Assista a esses vídeos antes de seguir.** Se a pose humana estiver errada, nada adiante conserta.

## 4. Etapa 2: humano → T1 (GMR)

O GMR (General Motion Retargeting, `github.com/YanjieZe/GMR`) ajusta o robô à pose humana frame a frame por cinemática inversa, respeitando os limites das juntas. Ele tem configuração pronta para o T1 com entrada SMPL-X.

### 4.1 Instalação

```bash
git clone https://github.com/YanjieZe/GMR.git
cd GMR
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
# se der problema de renderização:
conda install -c conda-forge libstdcxx-ng -y
```

Copie os modelos SMPL-X para dentro do GMR:

```
GMR/assets/body_models/smplx/
    SMPLX_NEUTRAL.pkl
    SMPLX_FEMALE.pkl
    SMPLX_MALE.pkl
```

### 4.2 Qual robô usar

| Nome no GMR | Modelo | Use? |
|---|---|---|
| `booster_t1` | `assets/booster_t1/T1_serial.xml`, 23 juntas | **Sim** |
| `booster_t1_29dof` | braços de 7 juntas | Não. O `booster_train` treina o T1 de 23 juntas |
| `booster_k1` | K1 | Só para o K1 |

A ordem das juntas do `T1_serial.xml` do GMR é a mesma de `T1_CSV_JOINT_NAMES`: cabeça, braço esquerdo, braço direito, cintura, perna esquerda, perna direita. Isso foi conferido junta a junta, e por isso não é preciso reordenar colunas. Os nomes diferem só na grafia: no GMR é `Left_Hip_Pitch`, no `booster_train` é `left_hip_pitch_joint`.

### 4.3 Execução

```bash
cd <GMR>
python scripts/gvhmr_to_robot.py \
    --gvhmr_pred_file <GVHMR>/outputs/demo/movimento/hmr4d_results.pt \
    --robot booster_t1 \
    --save_path <SAIDA>/movimento.pkl \
    --record_video
```

- **`--save_path` é necessário.** Sem ele, o script só mostra o movimento e não grava nada.
- `--record_video` grava um vídeo do robô. `--rate_limit` faz a visualização rodar na velocidade real.
- O `.pkl` gerado tem as chaves `fps`, `root_pos`, `root_rot` (já em `x, y, z, w`), `dof_pos`, `local_body_pos` e `link_body_list`.
- O GVHMR entrega o movimento a **30 fps**, e é esse o `fps` gravado no `.pkl`. Confira com:

  ```bash
  python -c "import pickle; d = pickle.load(open('<SAIDA>/movimento.pkl', 'rb')); print(d['fps'], d['dof_pos'].shape)"
  ```

Para rever o resultado depois:

```bash
python scripts/vis_robot_motion.py --robot booster_t1 --robot_motion_path <SAIDA>/movimento.pkl
```

## 5. Etapa 3: `.pkl` → CSV

O GMR traz um conversor no formato do BeyondMimic, que é o formato deste repositório:

```bash
python scripts/batch_gmr_pkl_to_csv.py --folder <SAIDA>
```

- Converte **todos** os `.pkl` da pasta e grava em `<SAIDA>/csv/<nome>.csv`.
- As colunas são `root_pos | root_rot | dof_pos`, sem cabeçalho, como pede a seção 1.
- Se o `fps` do `.pkl` passar de 30, o script descarta frames para chegar a cerca de 30 fps. Com a saída do GVHMR, que já é 30, ele não muda nada.
- O FPS não vai para o CSV. Anote-o para usar em `--input_fps`.

Se preferir não usar o script do GMR, a conversão é uma linha:

```python
import pickle, numpy as np
d = pickle.load(open("movimento.pkl", "rb"))
np.savetxt("movimento.csv", np.hstack([d["root_pos"], d["root_rot"], d["dof_pos"]]), delimiter=",", fmt="%.6f")
```

## 6. Etapa 4: limpar o CSV (recomendado)

Movimento vindo de vídeo monocular costuma ter três defeitos:

| Defeito | Efeito no treino | Correção |
|---|---|---|
| Tremedeira de frame a frame | Velocidades de referência altas e ruidosas; a recompensa de velocidade fica difícil de cumprir | Filtro Savitzky-Golay nas colunas |
| Raiz na altura errada (pés afundados ou flutuando) | O robô é inicializado dentro ou acima do chão a cada reinício | Deslocar `z` da raiz por uma constante |
| Começo e fim ruins | Poses estranhas enquanto a pessoa entra no quadro | Recortar frames |

O script do apêndice A faz as três coisas. Ele precisa de `numpy`, `scipy` e `mujoco` (`pip install mujoco scipy`) e não precisa do Isaac Lab:

```bash
python postprocess_csv.py \
    --input <SAIDA>/csv/movimento.csv \
    --output <SAIDA>/csv/movimento_limpo.csv \
    --mjcf <booster_assets>/robots/T1/T1_23dof.xml \
    --reference <booster_assets>/motions/T1/t1_dance.csv \
    --frame_range 30 420 \
    --window 9
```

- **Chão.** O script calcula a altura dos tornozelos em cada frame com o modelo MuJoCo do T1, pega a altura típica do pé de apoio (percentil 5 do pé mais baixo) e desloca a raiz para ela bater com a do movimento de referência `t1_dance.csv`, que já está correto. Sem `--reference`, a altura não é mexida.
- **Suavização.** `--window` é a janela do filtro, em frames, e tem que ser ímpar. A 30 fps, 9 frames são 0,3 s. Aumente se ainda tremer; diminua se o movimento perder velocidade, como no pico de um chute. Use `--window 0` para desligar.
- **Recorte.** `--frame_range` usa índices a partir de 1, inclusivos, como no `csv_to_npz.py`. Dá para recortar só no `csv_to_npz.py`, mas recortar aqui antes evita que a parte ruim influencie a suavização e a correção do chão.

## 7. Etapa 5: gerar o `.npz` e conferir

```bash
python scripts/csv_to_npz.py --headless --robot t1 \
    --input_file=<SAIDA>/csv/movimento_limpo.csv --input_fps=30 \
    --output_file=<SAIDA>/movimento.npz

python scripts/replay_npz.py --robot t1 --motion=<SAIDA>/movimento.npz
```

- **`--robot t1` é obrigatório nos dois scripts.** O padrão é `k1`.
- `--input_fps` é o FPS do CSV, 30 para a saída do GVHMR. Errar esse valor deixa o movimento mais rápido ou mais lento. `--output_fps` fica no padrão de 50.

No replay, confira:

- Os pés tocam o chão sem afundar e sem flutuar.
- A frente do robô aponta para onde a pessoa olhava, e ele não está deitado nem de cabeça para baixo. Se estiver, o quaternion foi lido na ordem errada.
- Cada membro se move como no vídeo. Braço esquerdo trocado com o direito indica ordem de juntas errada.
- Nenhuma junta salta de repente entre dois frames.

Depois aponte `motion_file` em `env_cfg.py` para o `.npz`, com caminho absoluto, e siga o [guia de uso](guia_de_uso.md), seção 2.3.

## 8. Pontos a confirmar na primeira execução

- **Origem da raiz.** O GMR grava a pose do corpo `Trunk` do `T1_serial.xml`, e o `booster_train` aplica essa pose ao `trunk` do `T1_23dof.urdf`. Os nomes batem, mas não foi conferido se as origens dos dois modelos coincidem. Uma diferença só de altura é absorvida pela correção de chão da seção 6; uma diferença de rotação aparece no replay como robô inclinado.
- **Limites das juntas.** O GMR respeita os limites do XML dele. Não foi conferido se são iguais aos do `T1_23dof.urdf`. Se o replay mostrar juntas batendo no limite, compare os dois arquivos.
- **Escala.** O GMR escala o humano para o tamanho do T1. Passos longos de uma pessoa alta podem virar passos que o T1 não alcança sem escorregar. Nesse caso, grave o movimento com passos mais curtos.
- **Cabeça.** O GMR gera ângulos para as duas juntas da cabeça a partir da cabeça humana. Se for treinar a variante sem cabeça (guia de uso, seção 5), esses valores ainda entram na observação, mas a política não os comanda.

## 9. Problemas comuns

| Sintoma | Causa provável |
|---|---|
| `CSV motion for T1 must have 30 columns` | Usou `booster_t1_29dof` no GMR, ou o CSV tem colunas a mais |
| `CSV joint columns do not match` | CSV com cabeçalho de outro robô; remova o cabeçalho ou corrija os nomes |
| Robô deitado ou girado no replay | Quaternion em `w, x, y, z`; reordene as colunas 3–6 para `x, y, z, w` |
| Movimento rápido ou lento demais | `--input_fps` diferente do FPS real do CSV |
| Pés atravessando o chão ou flutuando | Altura da raiz; rode o apêndice A com `--reference` |
| Robô tremendo no replay | Pose ruidosa do GVHMR; aumente `--window` ou grave de novo com câmera fixa e mais luz |
| Pés deslizando no chão | Limitação de vídeo monocular; reduza com câmera fixa (`-s`), pés sempre visíveis e passos mais curtos |
| `replay_npz.py` mostra outro robô | Faltou `--robot t1` |

## Apêndice A: `postprocess_csv.py`

Testado sobre o `t1_dance.csv` deslocado 10 cm para cima: o script corrigiu a altura em −0,100 m, e a suavização com janela 9 alterou as juntas em no máximo 0,011 rad.

```python
"""Limpa um CSV de movimento do T1 vindo do GMR: recorte, suavização e correção da altura do chão."""

import argparse

import mujoco
import numpy as np
from scipy.signal import savgol_filter

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def ankle_heights(model, data, motion):
    """Altura (z) dos dois tornozelos em cada frame."""
    ids = [model.body(name).id for name in FEET]
    z = np.empty((len(motion), 2))
    for i, row in enumerate(motion):
        data.qpos[:3] = row[:3]
        data.qpos[3:7] = row[[6, 3, 4, 5]]  # CSV xyzw -> MuJoCo wxyz
        data.qpos[7:] = row[7:]
        mujoco.mj_kinematics(model, data)
        z[i] = data.xpos[ids, 2]
    return z


def contact_height(model, data, motion):
    """Altura típica do tornozelo de apoio (percentil 5 do pé mais baixo)."""
    return np.percentile(ankle_heights(model, data, motion).min(axis=1), 5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="CSV gerado a partir do GMR")
    p.add_argument("--output", required=True)
    p.add_argument("--mjcf", required=True, help="booster_assets/robots/T1/T1_23dof.xml")
    p.add_argument("--reference", help="CSV de referência com o chão correto (ex.: t1_dance.csv)")
    p.add_argument("--frame_range", type=int, nargs=2, metavar=("INICIO", "FIM"),
                   help="Índices a partir de 1, inclusivos")
    p.add_argument("--window", type=int, default=9, help="Janela do Savitzky-Golay (ímpar); 0 desliga")
    args = p.parse_args()

    motion = np.loadtxt(args.input, delimiter=",")
    if args.frame_range:
        start, end = args.frame_range
        motion = motion[start - 1 : end]
    if motion.shape[1] != 30:
        raise SystemExit(f"Esperadas 30 colunas (7 raiz + 23 juntas), veio {motion.shape[1]}.")

    # Quaternions vizinhos no mesmo hemisfério, para a suavização não cruzar q e -q.
    for i in range(1, len(motion)):
        if np.dot(motion[i, 3:7], motion[i - 1, 3:7]) < 0:
            motion[i, 3:7] *= -1

    if args.window > 0:
        window = min(args.window, len(motion) - (1 - len(motion) % 2))
        motion = savgol_filter(motion, window_length=window, polyorder=3, axis=0)
        motion[:, 3:7] /= np.linalg.norm(motion[:, 3:7], axis=1, keepdims=True)

    if args.reference:
        model = mujoco.MjModel.from_xml_path(args.mjcf)
        data = mujoco.MjData(model)
        reference = np.loadtxt(args.reference, delimiter=",")
        offset = contact_height(model, data, reference) - contact_height(model, data, motion)
        motion[:, 2] += offset
        print(f"Altura da raiz corrigida em {offset:+.4f} m")

    np.savetxt(args.output, motion, delimiter=",", fmt="%.6f")
    print(f"{len(motion)} frames salvos em {args.output}")


if __name__ == "__main__":
    main()
```
