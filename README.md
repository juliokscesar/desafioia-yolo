# Desafio IA - Detecção de objetos usando YOLO

- [Detalhes](#detalhes)
- [Código e Reprodução]()
- [Resultados](#resultados)

## Detalhes
Utilizei um modelo treinado na plataforma [Roboflow](https://universe.roboflow.com/) com um dataset que é a junção de vários datasets de projetos de detecção de folhas com YOLO.

O modelo final, treinado a partir do modelo YOLO-NAS, pode ser encontrado [aqui](https://universe.roboflow.com/jc-98d3n/leaf-detection-h0stp/model/1).

Links das ferramentas principais que utilizei:
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [Ultralytics docs](https://docs.ultralytics.com/)
- Datasets e Modelos: [Roboflow](https://universe.roboflow.com/)
- [Supervision docs](https://supervision.roboflow.com/latest/)
- [SAHI](https://github.com/obss/sahi)

Os meus rascunhos e testes de códigos estão todos no Jupyter Notebook [rascunhos_desafio.ipynb].


## Código e reprodução
Para utilizar o código de detecção, é necessário clonar o repositório e instalar todos os pacotes que o código utiliza (recomendo ser em um ambiente virtual):
```
git clone ...
cd ...
python -m venv .venv

source .venv/bin/activate # Linux
.venv\Scripts\activate # Windows

pip install -r requirements.txt
```

**IMPORTANTE**: É necessário ter uma chave de API da plataforma Roboflow. É só ter uma conta na plataforma e seguir para Configurações da Conta > Roboflow API e copiar a "Private API Key". Então é só setar a chave como uma variável de ambiente com nome `ROBOFLOW_API_KEY` antes de executar a ferramenta:
- Linux (Shell): `export ROBOFLOW_API_KEY="chave"`
- Windows (PowerShell): `$Env:ROBOFLOW_API_KEY = "chave"`

E então é só executar `python detect.py [args]`, que funcionará como uma ferramenta no terminal. Executando `python detect.py -h` explica as opções para os argumentos:
```
usage: desafio.py [-h] [-s] [--confidence CONFIDENCE] [--overlap OVERLAP] [--slicew SLICEW] [--sliceh SLICEH]
                  [--sliceoverlapw SLICEOVERLAPW] [--sliceoverlaph SLICEOVERLAPH]
                  image

Desafio de deteccao de objetos com YOLO

positional arguments:
  image                 Imagem para fazer a detecção

options:
  -h, --help            show this help message and exit
  -s, --save            Salvar a imagem com detecções ao invés de plotar
  --confidence CONFIDENCE, -c CONFIDENCE
                        Valor 'confidence' para fazer a inferencia (entre 0 e 100)
  --overlap OVERLAP, -o OVERLAP
                        Valor máximo de 'overlap' até juntar duas detecções (entre 0 e 100)
  --slicew SLICEW       Largura de cada recorte (slice) da imagem para fazer inferências
  --sliceh SLICEH       Altura de cada recorte (slice) da imagem para fazer inferência
  --sliceoverlapw SLICEOVERLAPW
                        Valor de largura de 'overlap' para cada divisão (slice) (entre 0 e 1)
  --sliceoverlaph SLICEOVERLAPH
                        Valor de altura de 'overlap' para cada divisão (slice) (entre 0 e 1)
```

As imagens do desafio estão no diretório `imgs/`, então para usar a ferramenta e detectar e contar as folhas da imagem 0.jpg, por exemplo:
```
python detect.py imgs/0.jpg
```

A ferramenta então mostra quantos objetos na imagem e abre um plot (do matplotlib) mostrando a imagem com as caixas delimitando os objetos. 

Caso queira salvar a imagem resultante localmente, é só passar a opção `--save`:
```
python detect.py imgs/0.jpg --save
```

A imagem será salva no diretório `exp/` com um nome aleatório.

### Descrição do código
O código basicamente utiliza a API da plataforma Roboflow pelos pacotes `roboflow` e `supervision` para poder utilizar o modelo disponível nela e fazer as inferências, já que não foi possível treinar o modelo na minha própria máquina. Então tem duas funções principais: 
- `slice_infer_image`: utiliza a framework SAHI embutida na biblioteca `supervision` para fazer inferências em recortes da imagem, melhorando a detecção de objetos pequenos.

- `generate_annotated_image`: só pega os resultados das detecções da função `slice_infer_image` e gera uma nova imagem com as anotações das caixas que delimitam os objetos detectados.


## Resultados

