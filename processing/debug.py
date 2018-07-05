import utils
import os


uid = "1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860"
images = utils.load_patient_images(uid, "_i.png")
cube_img = utils.get_cube_from_img(images, 100, 100, 100, 64)
target_path = os.path.join("/home/alyb/ConvNetDiagnosis/processing/",uid + "test.png")
utils.save_cube_img(target_path, cube_img, 8, 8)
volume = utils.load_cube_img(target_path, 8, 8, 64)


