#!/usr/bin/env python

import sys, os, json, argparse, collections
import numpy as np
import DeepFried2 as df

from examples.utils import imread, imresize


def resizecropto(img, size=224):
    # First, resize the smallest side to `size`.
    img = imresize(img, h=img.shape[0]*size//min(img.shape[:2]),
                        w=img.shape[1]*size//min(img.shape[:2]))

    # Then, crop-out the central part of the largest side.
    return img[(img.shape[0]-size)//2:img.shape[0]-(img.shape[0]-size)//2,
               (img.shape[1]-size)//2:img.shape[1]-(img.shape[1]-size)//2,
               :]


def here(fname):
    return os.path.join(os.path.dirname(__file__), fname)


def printnow(fmt, *a, **kw):
    sys.stdout.write(fmt.format(*a, **kw))
    sys.stdout.flush()


if __name__ == "__main__":
    if __package__ is None:  # PEP366
        __package__ = "DeepFried2.examples.Pretrained"

    parser = argparse.ArgumentParser(description="Runs the image through a model pre-trained on ImageNet.")
    parser.add_argument('-m', '--model', choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument('-r', '--raw', action='store_true', help="Do not subtract the trainset's channel-mean")
    parser.add_argument('image', nargs='?')
    args = parser.parse_args()

    if args.model == 'vgg16':
        vgg = df.zoo.vgg16
    elif args.model == 'vgg19':
        vgg = df.zoo.vgg19

    printnow("Loading models...")
    modelFC, mean, classes = vgg.pretrained(fully_conv=True)
    model, mean, classes = vgg.pretrained(fully_conv=False)
    printnow("Done\n")

    # Switch to prediction mode for deterministic output.
    # NOTE: it can be interesting[1] to evaluate in training mode!
    # 1: http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html
    modelFC.evaluate()
    model.evaluate()

    # Potentially download example image from gist.
    args.image = args.image or df.zoo.download('https://gist.github.com/lucasb-eyer/237d03b9fcdb0fe03bce/raw/ILSVRC2012_val_00000151.JPEG')

    printnow("Loading and preparing image...")
    img = imread(args.image)

    # Subtracting the per-channel mean (stored in `mean`) as only pre-processing.
    if not args.raw:
        img -= mean

    # For the regular network, we need to resize and crop the image.
    img224 = resizecropto(img, 224)

    # Get them from disk-space HWC to nnet-space CHW.
    img = img.transpose((2,0,1))
    img224 = img224.transpose((2,0,1))

    # Put them into a lonely minibatch.
    minibatch = np.array([img])
    minibatch224 = np.array([img224])
    printnow("Done\n")

    # And send them through the networks.
    printnow("Regular model:\n")
    preds = np.array(model.forward(minibatch224))
    top1 = np.argmax(preds[0])
    printnow("  - shape of predictions: {}\n", preds.shape)
    printnow("  - top-1 prediction: {}\n", top1)

    printnow("Fully-conv model:\n")
    predsFC = np.array(modelFC.forward(minibatch))
    top1FC = np.argmax(predsFC[0], axis=0)
    printnow("  - shape of predictions: {}\n", predsFC.shape)
    printnow("  - top-1 of average: {}\n", np.argmax(np.mean(predsFC, axis=(2,3))[0]))
    printnow("  - top-1 map:\n{}\n", top1FC)

    # Now, we want to print the relevant words that occured as predictions.
    ilsvrc_words = json.load(open(df.zoo.download('https://gist.github.com/lucasb-eyer/237d03b9fcdb0fe03bce/raw/ILSVRC2012_words.json')))
    ilsvrc_ids   = json.load(open(df.zoo.download('https://gist.github.com/lucasb-eyer/237d03b9fcdb0fe03bce/raw/ILSVRC2012_ids.json')))

    for i, _ in collections.Counter([top1] + list(top1FC.flat)).most_common():
        printnow('{} (wnid: {}, ilsvrcid: {}): {}\n', i, classes[i], ilsvrc_ids[classes[i]], ilsvrc_words[classes[i]])
