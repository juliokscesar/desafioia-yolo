import argparse
import os
from roboflow import Roboflow
import supervision as sv
import numpy as np
import cv2
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt


def slice_infer_image(api_key: str, imgpath: str, conf=40, overlap=20, slice_wh=(480, 480), slice_overlap_ratio=(0.1, 0.1)) -> sv.Detections:    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("leaf-detection-a4rgd")
    model = project.version(5).model

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
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()

    default_img = cv2.imread(default_imgpath)

    annotated_image = box_annotator.annotate(
        scene=default_img.copy(),
        detections=detections
    )

    annotated_image = label_annotator.annotate(
        scene=annotated_image,
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


def save_image(img: np.ndarray, dir: str):
    with sv.ImageSink(target_dir_path=dir) as sink:
        sink.save_image(image=img)


def main():
    # Main parser
    parser = argparse.ArgumentParser(
        description="Desafio de deteccao de objetos com YOLO"
    )

    parser.add_argument("image", help="Imagem para fazer a detecção")
    parser.add_argument("-s", "--save", action="store_true", help="Salvar a imagem com detecções ao invés de plotar")

    parser.add_argument("--confidence", "-c", 
                        type=int, 
                        nargs=1, 
                        default=40, 
                        help="Valor 'confidence' para fazer a inferencia (entre 0 e 100)"
    )
    parser.add_argument("--overlap", "-o",
                        type=int,
                        nargs=1,
                        default=20,
                        help="Valor máximo de 'overlap' até juntar duas detecções (entre 0 e 100)"
    )

    parser.add_argument("--slicew",
                        type=int,
                        nargs=1,
                        default=480,
                        help="Largura de cada recorte (slice) da imagem para fazer inferências"
    )
    parser.add_argument("--sliceh",
                        type=int,
                        nargs=1,
                        default=480,
                        help="Altura de cada recorte (slice) da imagem para fazer inferência"
    )

    parser.add_argument("--sliceoverlapw",
                        type=float,
                        nargs=1,
                        default=0.1,
                        help="Valor de largura de 'overlap' para cada divisão (slice) (entre 0 e 1)"
    )
    parser.add_argument("--sliceoverlaph",
                        type=float,
                        nargs=1,
                        default=0.1,
                        help="Valor de altura de 'overlap' para cada divisão (slice) (entre 0 e 1)"
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

    print(f"Contagem total: {len(detections.xyxy)} objetos")

    if args.save:
       save_image(annotated_image, "exp")
    else:
        plot_image(annotated_image)



if __name__ == "__main__":
    main()
