import cv2
import glob
import tqdm


def full_fname_to_core_name(img_fname):
    """ hehe/biz.png -> biz """
    return img_fname.split('/')[-1].split('.')[0]


def show_transformation_on_test_images(transformation, name):
    """
    Loop over test images 
    """
    for img_fname in tqdm.tqdm(glob.glob('img/*')):
        img = cv2.imread(img_fname)
        fname = full_fname_to_core_name(img_fname)
        post_img = transformation(img)
        cv2.imwrite('out/' + name + '_{}.png'.format(fname), post_img)
