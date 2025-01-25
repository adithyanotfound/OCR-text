from flask import Flask, request, jsonify
import PyPDF2
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

app = Flask(__name__)

def pdf_to_text_with_graphics(pdf_path, output_txt_path, image_output_dir):
    try:
        if not os.path.exists(pdf_path):
            return f"Error: The file '{pdf_path}' does not exist.", None

        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

        doc = fitz.open(pdf_path)
        extracted_text = ""

        for page_num, page in enumerate(doc):
            # Extract text from the page
            extracted_text += page.get_text()

            # Extract images from the page
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image = Image.open(io.BytesIO(image_bytes))
                
                # Save the image
                image_filename = os.path.join(image_output_dir, f"page{page_num + 1}_img{img_index + 1}.{image_ext}")
                image.save(image_filename)

                # Perform OCR on the image
                ocr_text = pytesseract.image_to_string(image)
                extracted_text += f"\n[Image Text from {image_filename}]:\n{ocr_text}\n"

        # Save extracted text to a file
        with open(output_txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(extracted_text)

        return f"Text and image data extracted and saved to '{output_txt_path}'.", extracted_text

    except Exception as e:
        return f"An error occurred: {e}", None

def pdf_to_text(pdf_path, output_txt_path):
    try:
        # Check if the PDF file exists
        if not os.path.exists(pdf_path):
            return f"Error: The file '{pdf_path}' does not exist.", None

        # Open the PDF file
        with open(pdf_path, 'rb') as pdf_file:
            # Create a PDF reader object
            reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page in reader.pages:
                text += page.extract_text()

        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)

        return f"Text extracted and saved to '{output_txt_path}'.", text

    except Exception as e:
        return f"An error occurred: {e}", None

@app.route('/extract', methods=['POST'])
def extract_pdf():
    try:
        pdf_file = request.files['pdf']
        output_txt_path = "output_with_graphics.txt"
        image_output_dir = "extracted_images"

        # Save the uploaded PDF file
        pdf_path = os.path.join("uploads", pdf_file.filename)
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        pdf_file.save(pdf_path)

        # Process the PDF
        message, extracted_text = pdf_to_text_with_graphics(pdf_path, output_txt_path, image_output_dir)

        if extracted_text:
            return jsonify({"message": message, "extracted_text": extracted_text}), 200
        else:
            return jsonify({"error": message}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    app.run(debug=True)
