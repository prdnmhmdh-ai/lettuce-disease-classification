import os
import base64
import requests
from flask import Flask, send_from_directory, render_template, request, jsonify
from inference_sdk import InferenceHTTPClient
from PIL import Image
from io import BytesIO
from roboflow import Roboflow
import supervision as sv
import cv2
import traceback
import numpy as np

app = Flask(__name__, static_folder='assets')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max size for uploaded images

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="zHXvPR4wY6HIpFlHqkFg"
)

# Pastikan folder untuk upload dan result ada
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Fungsi untuk upload file ke Vercel Blob
def upload_to_vercel_blob(file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()

    # Gantilah dengan token API Vercel kamu
    headers = {
        "Authorization": "Bearer YOUR_VERCEL_API_TOKEN",  # Ganti dengan token API Vercel kamu
    }

    url = "https://api.vercel.com/v1/storage/your_project_name/files"  # Sesuaikan dengan URL Vercel API kamu
    files = {'file': (os.path.basename(file_path), file_data)}
    
    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        return response.json()['url']  # Mengembalikan URL file di Vercel Blob
    else:
        return None

@app.route("/static/results/<path:filename>")
def serve_result_image(filename):
    return send_from_directory('static/results', filename)

@app.route("/static/uploads/<path:filename>")
def serve_upload_image(filename):
    return send_from_directory('static/uploads', filename)

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(os.getcwd(), "index.html")

@app.route("/", methods=["POST"])
def detect():
    try:
        # Terima gambar dari kamera atau upload
        data = request.get_json()
        if 'camera_image' in data:
            data_url = data['camera_image']
            header, encoded = data_url.split(",", 1)
            binary_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(binary_data))

            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'camera_capture.jpg')
            image.save(image_path, optimize=True, quality=75)

        elif 'image' in request.files:
            file = request.files['image']
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

        else:
            return jsonify({"reply": "❌ No image provided"}), 400

        print("📦 Loading Roboflow...")
        rf = Roboflow(api_key="zHXvPR4wY6HIpFlHqkFg")
        project = rf.workspace().project("aquaponic_polygan_disease_test")
        model = project.version(5).model

        print("🔍 Predicting...")
        result_json = model.predict(image_path, confidence=40).json()
        preds = result_json["predictions"]

        if not preds:
            return jsonify({
                "predictions": [],
                "annotated_image": None,
                "reply": "✅ Tidak ada objek yang terdeteksi."
            })

        # Konversi prediksi ke objek Detections
        xyxy = np.array([ 
            [pred["x"] - pred["width"] / 2, 
             pred["y"] - pred["height"] / 2, 
             pred["x"] + pred["width"] / 2, 
             pred["y"] + pred["height"] / 2]
            for pred in preds
        ])
        class_id = np.array([pred.get("class_id", 0) for pred in preds])
        confidence = np.array([pred["confidence"] for pred in preds])
        class_name = [pred["class"] for pred in preds]

        detections = sv.Detections(
            xyxy=xyxy,
            class_id=class_id,
            confidence=confidence,
            data={"class_name": class_name}
        )

        # Tambahkan label ke data
        detections.data["class_name"] = [f"{c} ({conf:.2f})" for c, conf in zip(class_name, confidence)]

        # Baca gambar dan anotasi
        image_cv2 = cv2.imread(image_path)
        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator()
        labels = [f"{c} ({conf:.2f})" for c, conf in zip(class_name, confidence)]

        image_with_box = box_annotator.annotate(
            scene=image_cv2,
            detections=detections
        )

        annotated_image = label_annotator.annotate(
            scene=image_with_box,
            detections=detections,
            labels=labels
        )

        # Simpan hasil anotasi
        output_filename = "annotated_" + os.path.basename(image_path)
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        success = cv2.imwrite(output_path, annotated_image)

        if not success:
            return jsonify({"reply": "❌ Gagal menyimpan hasil gambar."}), 500

        # Upload hasil ke Vercel Blob
        image_url = upload_to_vercel_blob(output_path)

        if image_url:
            return jsonify({
                "predictions": preds,
                "annotated_image": image_url,
                "reply": "✅ Gambar berhasil di-upload ke Vercel Blob!"
            })
        else:
            return jsonify({"reply": "❌ Gagal meng-upload gambar ke Vercel Blob."}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": f"❌ Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
