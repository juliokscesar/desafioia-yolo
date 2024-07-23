import argparse
import os
from roboflow import Roboflow
import supervision as sv
import numpy as np
import cv2
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt


def get_roboflow_model(api_key: str, project: str, version: int):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project)
    model = project.version(version).model

    return model


def slice_infer_image(api_key: str, imgpath: str, conf=50, overlap=50, slice_wh=(640, 640), slice_overlap_ratio=(0.1, 0.1)) -> sv.Detections:    
    #model = get_roboflow_model(api_key, "plant-leaf-detection-att1p", 1)
    model = get_roboflow_model(api_key, "plant-leaf-detection-att1p", 2)

    def sv_slice_callback(image: np.ndarray) -> sv.Detections:
        with tempfile.NamedTemporaryFile(suffix=Path(imgpath).suffix) as f:
            cv2.imwrite(f.name, image)
            result = model.predict(f.name, confidence=conf, overlap=overlap).json()

        detections = sv.Detections.from_inference(result)
        return detections
    
    image = cv2.imread(imgpath)

    slicer = sv.InferenceSlicer(callback=sv_slice_callback, slice_wh=slice_wh, overlap_ratio_wh=slice_overlap_ratio)
    sliced_detections = slicer(image=image)

    return sliced_detections


def generate_annotated_image(default_imgpath: str, detections: sv.Detections) -> np.ndarray:
    box_annotator = sv.BoxAnnotator(thickness=1)

    default_img = cv2.imread(default_imgpath)

    annotated_image = box_annotator.annotate(
        scene=default_img.copy(),
        detections=detections
    )

    return annotated_image


def plot_image(img: np.ndarray):
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def save_image(img: np.ndarray, dir: str, name: str):
    with sv.ImageSink(target_dir_path=dir) as sink:
        sink.save_image(image=img, image_name=name)
    
    print(f"Imagem salva em ./{dir}/{name}")


def main():
    # Main parser
    parser = argparse.ArgumentParser(
        description="Desafio de deteccao de objetos com YOLO"
    )

    parser.add_argument("image", help="Imagem para fazer a detecção")
    parser.add_argument("-s", "--save", action="store_true", help="Salvar a imagem com detecções ao invés de plotar")

    parser.add_argument("--confidence", "-c", 
                        type=int, 
                        default=50, 
                        help="Padrão: 50. Valor 'confidence' para fazer a inferencia (entre 0 e 100). Objetos detectados com valor abaixo desse serão descartados."
    )
    parser.add_argument("--overlap", "-o",
                        type=int,
                        default=50,
                        help="Padrão: 50. Valor máximo de 'overlap' até juntar duas detecções (entre 0 e 100)"
    )

    parser.add_argument("--slicew",
                        type=int,
                        default=640,
                        help="Padrão: 640. Largura de cada recorte (slice) da imagem para fazer inferências"
    )
    parser.add_argument("--sliceh",
                        type=int,
                        default=640,
                        help="Padrão: 640. Altura de cada recorte (slice) da imagem para fazer inferência"
    )

    parser.add_argument("--sliceoverlapw",
                        type=float,
                        default=0.1,
                        help="Padrão: 0.1. Valor de largura de 'overlap' para cada divisão (slice) (entre 0 e 1)"
    )
    parser.add_argument("--sliceoverlaph",
                        type=float,
                        default=0.1,
                        help="Padrão: 0.1. Valor de altura de 'overlap' para cada divisão (slice) (entre 0 e 1)"
    )

    args = parser.parse_args()

    img_path = args.image
    if not(Path(img_path).is_file()):
        print(f"Não foi possível encontrar {img_path}")
        return
    
    conf = args.confidence
    overlap = args.overlap
    slice_wh = (args.slicew, args.sliceh)
    slice_overlap_wh = (args.sliceoverlapw, args.sliceoverlaph)


    api_key = ""
    try:
        api_key = os.environ["ROBOFLOW_API_KEY"]
    except:
        print("É necessário setar a chave de API do Roboflow como variável de ambiente 'ROBOFLOW_API_KEY'")
        return

    detections = slice_infer_image(api_key, img_path, conf, overlap, slice_wh, slice_overlap_wh)
    annotated_image = generate_annotated_image(img_path, detections)

    print(f"Parâmetros: Conf={conf}, overlap={overlap}, slice_wh={slice_wh}, slice_overlap_wh={slice_overlap_wh}")
    print(f"Contagem total: {len(detections.xyxy)} objetos")

    if args.save:
       save_image(annotated_image, "exp", "exp" + os.path.basename(img_path))
    else:
        plot_image(annotated_image)



if __name__ == "__main__":
    main()
