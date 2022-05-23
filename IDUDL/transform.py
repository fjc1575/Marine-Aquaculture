from skimage import io, img_as_float
import numpy as np

def gamma(img):
    Cimg = img/255
    ga = 1.5
    out = np.power(Cimg, ga)

    return out

def truncated_linear_stretch(image, truncated_value, max_out=255, min_out=0):

    def gray_process(gray):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        if (max_out <= 255):
            gray = np.uint8(gray)
        elif (max_out <= 65535):
            gray = np.uint16(gray)
        return gray

    #  如果是多波段
    if (len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
    #  如果是单波段
    else:
        image_stretch = gray_process(image)

    return image_stretch

def Gaussian(img):
    mean = 0
    var = 0.01
    image = img_as_float(img)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy

def trans(img):

    img_roi = img

    out_Gaussian = Gaussian(img_roi)

    out_turn5 = truncated_linear_stretch(img_roi, 5)
    out_turn7 = truncated_linear_stretch(img_roi, 7)
    out_turn2 = truncated_linear_stretch(img_roi, 2)

    out_gam = gamma(img_roi)

    com = np.array([out_Gaussian, out_turn5, out_turn7, out_turn2, out_gam])

    return com

