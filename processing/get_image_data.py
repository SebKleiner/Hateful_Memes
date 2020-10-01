import io
from google.cloud import vision
import os

password_file = "write here yours"  # Write here yours
# path = 'uploads/01235.png'


def get_all(path, password_file=password_file):
    # Set up google vision
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = password_file
    client = vision.ImageAnnotatorClient()

    # Read image
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations
    ret_text = [text.description for text in texts]
    print(ret_text)

    # Label detection
    response = client.label_detection(image=image)
    labels = response.label_annotations
    ret_label = [label.description for label in labels]
    print(ret_label)

    # Object detection
    objects = client.object_localization(
        image=image).localized_object_annotations
    ret_object = [(object_.name, object_.score) for object_ in objects]
    print(ret_object)

    return {'path': path, 'text': ret_text, 'labels': ret_label, 'objects': ret_object}


def get_test_dict():
    path = '48260.png'
    ret_text = ['woohooo pedal faster patrick,\nthe owners are coming\nasf',
                'pedal',
                'faster',
                'patrick,',
                'the',
                'owners',
                'are',
                'coming']
    ret_label = ['Vehicle',
                 'Photo caption',
                 'Bicycle wheel',
                 'Bicycle',
                 'Bicycle tire',
                 'Cool',
                 'Font',
                 'Photography',
                 'Adaptation',
                 'Bicycle frame']
    ret_object = [('Bicycle wheel', 0.8897398114204407),
                  ('Person', 0.8209506273269653),
                  ('Bicycle wheel', 0.7898780107498169),
                  ('Jeans', 0.7539083957672119),
                  ('Bicycle wheel', 0.689781129360199),
                  ('Outerwear', 0.6768908500671387),
                  ('Bicycle', 0.6365995407104492),
                  ('Glasses', 0.6281065940856934),
                  ('Pants', 0.6200207471847534),
                  ('Hat', 0.5796813368797302)]
    dicti = {'path': path, 'text': ret_text, 'labels': ret_label, 'objects': ret_object}
    return dicti