import cv2
import streamlit as st
import numpy as np


class SketchGenerator:
    def __init__(self, image):
        self.image = image

    def _invert_image(self, image):
        return 255 - image

    def _apply_gaussian_blur(self, image):
        return cv2.GaussianBlur(image, (21, 21), 0)

    def generate_sketch(self):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Invert the grayscale image
        inverted_gray_image = self._invert_image(gray_image)

        # Apply Gaussian Blur to the inverted image
        blurred_image = self._apply_gaussian_blur(inverted_gray_image)

        # Invert the blurred image
        inverted_blurred_image = self._invert_image(blurred_image)

        # Create the sketch by blending the inverted blurred image with the original grayscale image
        sketch = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)

        return sketch


def main():
    st.title("Sketch Generator")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Convert uploaded file to opencv image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Create a SketchGenerator object
        sketch_generator = SketchGenerator(opencv_image)

        # Generate the sketch
        sketch = sketch_generator.generate_sketch()

        # Display the sketch image
        st.image(sketch, caption="Sketch.", use_column_width=True)


if __name__ == "__main__":
    main()
