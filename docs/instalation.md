Instale as dependências do sistema:
bash

sudo apt update

sudo apt install -y \
    git \
    cmake \
    build-essential \
    python3.12 \
    python3.12-venv \
    python3.12-dev

2. Baixar o Isaac Lab

Clone o repositório oficial:
bash

git clone https://github.com/isaac-sim/IsaacLab.git \
    --branch release/3.0.0

cd IsaacLab

3. Criar o ambiente virtual Python

Dentro da pasta IsaacLab, execute:
bash

python3.12 -m venv env_isaaclab

Ative o ambiente:
bash

source env_isaaclab/bin/activate

Atualize o pip:
bash

python -m pip install --upgrade pip

Confira se está utilizando o Python correto:
bash

python --version
which python

O resultado deve indicar Python 3.12 e o executável dentro de env_isaaclab.
4. Instalar o Isaac Lab

Com o ambiente virtual ativado, execute:
bash

./isaaclab.sh -i


uv pip install "isaacsim[all,extscache]==6.2.0.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow
