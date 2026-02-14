from fast_alpr import ALPR

# Initilise ALPR 
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model",
)

# alpr_results = alpr.predict("test_img/img1.png")
# print("start", list(alpr_results))


def get_carplate(image):
    results = alpr.predict(image)

    if not results:
        return None

    r = results[0]

    return {
        "plate": r.ocr.text,
        "ocr_confidence": r.ocr.confidence,
        "detection_confidence": r.detection.confidence,
        "bbox": (
            r.detection.bounding_box.x1,
            r.detection.bounding_box.y1,
            r.detection.bounding_box.x2,
            r.detection.bounding_box.y2,
        )
    }