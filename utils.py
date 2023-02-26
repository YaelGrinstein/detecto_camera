import cv2
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
def read_image(path):
    """Helper function that reads in an image as a
    `NumPy <https://numpy.org/>`_ array. Equivalent to using
    `OpenCV <https://docs.opencv.org/master/>`_'s cv2.imread
    function and converting from BGR to RGB format.

    :param path: The path to the image.
    :type path: str
    :return: Image in NumPy array format
    :rtype: ndarray

    **Example**::

        >>> import matplotlib.pyplot as plt
        >>> from detecto.utils import read_image

        >>> image = read_image('image.jpg')
        >>> plt.imshow(image)
        >>> plt.show()
    """
    if not os.path.isfile(path):
        raise ValueError(f'Could not read image {path}')

    image = cv2.imread(path)

    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        raise ValueError(f'Could not convert image color: {str(e)}')

    return rgb_image


def _is_iterable(variable):
    return isinstance(variable, list) or isinstance(variable, tuple)



def show_labeled_image(image, boxes, labels=None,scores=None, idx=None, output_path=None):
    """Show the image along with the specified boxes around detected objects.
    Also displays each box's label if a list of labels is provided.

    :param image: The image to plot. If the image is a normalized
        torch.Tensor object, it will automatically be reverse-normalized
        and converted to a PIL image for plotting.
    :type image: numpy.ndarray or torch.Tensor
    :param boxes: A torch tensor of size (N, 4) where N is the number
        of boxes to plot, or simply size 4 if N is 1.
    :type boxes: torch.Tensor
    :param labels: (Optional) A list of size N giving the labels of
            each box (labels[i] corresponds to boxes[i]). Defaults to None.
    :type labels: torch.Tensor or None

    **Example**::

        >>> from detecto.core import Model
        >>> from detecto.utils import read_image
        >>> from detecto.visualize import show_labeled_image

        >>> model = Model.load('model_weights.pth', ['tick', 'gate'])
        >>> image = read_image('image.jpg')
        >>> labels, boxes, scores = model.predict(image)
        >>> show_labeled_image(image, boxes, labels)
    """
    cv2.imwrite('new_img.jpg', image)

    fig, ax = plt.subplots(1)
    # If the image is already a tensor, convert it back to a PILImage
    # and reverse normalize it
    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = transforms.ToPILImage()(image)
        cv2.imshow("img", image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image, cmap=plt.cm.jet)
    cv2.imwrite('my.jpg', image)


    if boxes.ndim == 1:
        boxes = boxes.view(1, 4)

    if labels is not None and not _is_iterable(labels):
        labels = [labels]
        scores = [scores]

    # Plot each box
    fontScale = 1.5
    for i in range(boxes.shape[0]):
        box = boxes[i]
        color = (0, 0, 255)
        width, height = (box[2] - box[0]).item(), (box[3] - box[1]).item()
        initial_pos = (box[0].item(), box[1].item())
        #pt2 = (box[2].item(), box[3].item())
        #cv2.rectangle(image, initial_pos, pt2, (255, 0, 0), thickness=2)
        rect = patches.Rectangle(initial_pos,  width, height, linewidth=1,
                                 edgecolor='r', facecolor='none')
        org = (box[0] + 5, box[1] - 5)
        text = '{}'.format(labels[i], scores[i])
        if labels:
            #image = cv2.putText(image, text, org , fontScale, color, cv2.LINE_AA)
            ax.text(box[0] + 5, box[1] - 5, '{}'.format(labels[i], scores[i]), color='red')

        ax.add_patch(rect)
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #cv2.imwrite(os.path.join(output_path, idx + "_img.png"), image)
    fig.savefig(os.path.join(output_path, idx + "_img.png"))




def reverse_normalize(image):
    """Reverses the normalization applied on an image by the
    :func:`detecto.utils.reverse_normalize` transformation. The image
    must be a `torch.Tensor <https://pytorch.org/docs/stable/tensors.html>`_
    object.

    :param image: A normalized image.
    :type image: torch.Tensor
    :return: The image with the normalization undone.
    :rtype: torch.Tensor


    **Example**::

        >>> import matplotlib.pyplot as plt
        >>> from torchvision import transforms
        >>> from detecto.utils import read_image, \\
        >>>     default_transforms, reverse_normalize

        >>> image = read_image('image.jpg')
        >>> defaults = default_transforms()
        >>> image = defaults(image)

        >>> image = reverse_normalize(image)
        >>> image = transforms.ToPILImage()(image)
        >>> plt.imshow(image)
        >>> plt.show()
    """

    reverse = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                   std=[1 / 0.229, 1 / 0.224, 1 / 0.255])
    return reverse(image)
