import os
import face_recognition


DIRECTORY = os.getcwd()


def get_images(name_dir):  # получаем картинки!
    for path, _, files in os.walk(os.path.join(DIRECTORY, name_dir)):
        for file in files:
            yield os.path.join(path, file)
            if name_dir == 'known_img':
                break


img_start = [*get_images('known_img')][0]
img_known = face_recognition.load_image_file(img_start)
unknown_image_list = get_images('unknown_png')

for un_image in unknown_image_list:
    encoding_biden = face_recognition.face_encodings(img_known)[0]
    encoding_unknown = face_recognition.face_encodings(un_image)[0]
    print(face_recognition.compare_faces([encoding_biden], encoding_unknown))

