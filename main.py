import cv2
import streamlit as st
import time
import torch

from PIL import Image

from detection import Detection


class Main:
    model = torch.hub.load('', 'custom', source='local', path='hamburger_menu_recognition.onnx', force_reload=True)
    model('test_img.jpg')  # start model

    st.title('Finder of hamburger menu on a screenshot')
    activities = ['Hamburger menu detection']
    choice = st.sidebar.selectbox('Select Activity', activities)

    if choice == 'Hamburger menu detection':
        st.subheader('Upload photo of website (jpg/png/jpeg)')

        uploaded_files = st.file_uploader('Upload images', type=['jpg', 'png', 'jpeg'],
                                          accept_multiple_files=True)  # file uploader

        st.write("Progress bar")
        progress_bar = st.progress(0)
        bar_counter = 0

        for uploaded_file in uploaded_files:  # iteration over uploaded files
            new_image = Image.open(uploaded_file)  # open image
            new_image.save(uploaded_file.name)  # save
            start_time = time.time()  # starting measuring detection time
            result_img = Detection(cv2.cvtColor(cv2.imread(uploaded_file.name), cv2.COLOR_BGR2RGB), model)  # detection
            result = result_img.do_detection()  # results of detection
            end_time = time.time()  # ending measuring detection time
            time_taken = end_time - start_time  # calculate measuring detection time
            if result is None:
                st.image(new_image)
                st.text('Nothing found')  # Text if nothing found
            else:
                st.image(result)
                st.text(f'Time taken: {round(time_taken * 1000, 5)} ms')

            bar_counter += 1 / len(uploaded_files)
            if bar_counter > 1:
                bar_counter = 1.0
            progress_bar.progress(bar_counter)


if __name__ == '__main__':
    website = Main()
