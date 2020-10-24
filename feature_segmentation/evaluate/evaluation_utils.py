
class Evaluation():

    def __init__(self, params, filename, model, mode, choroid):
        self.params = params
        self.model_dir = params.model_directory
        self.mode = mode
        self.model = model
        self.model_input_shape = (1, params.img_shape, params.img_shape, 3)
        self.filename = filename
        self.image, self.label = self.__load_test_image()
        self.prediction = self.__predict_image()
        self.seg_cmap, self.seg_norm, self.bounds = color_mappings()
        self.jaccard = jaccard_score(self.label.flatten(), self.prediction.flatten(), average=None)
        self.choroid = choroid

    def resize(self, im):
        desired_size = self.params.img_shape
        im = Image.fromarray(im)

        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        im = im.resize(new_size, Image.NEAREST)
        # create a new image and paste the resized on it

        new_im = Image.new("L", (desired_size, desired_size))
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))

        return np.array(new_im)

    def __load_test_image(self):
        # load samples
        im = Image.open(os.path.join(self.params.data_path, "images", self.filename))

        if self.params.choroid_latest:
            lbl = Image.open(os.path.join(self.params.data_path, "masks_choroid", self.filename))
        else:
            lbl = Image.open(os.path.join(self.params.data_path, "masks", self.filename))

        im = np.array(im)
        lbl = np.array(lbl)

        # resize samples
        im_resized = self.resize(im)

        # if image grey scale, make 3 channel
        if len(im_resized.shape) == 2:
            im_resized = np.stack((im_resized,) * 3, axis=-1)

        lbl_resized = self.resize(lbl)

        # convert choroid to background
        # lbl_resized[lbl_resized == 10] = 0

        im_scaled = np.divide(im_resized, 255., dtype=np.float32)

        self.image = im_resized
        return im_scaled, lbl_resized.astype(int)

    def __predict_image(self):
        # get probability map
        pred = self.model.predict(self.image.reshape(self.model_input_shape))

        return np.argmax(pred, -1)[0, :, :].astype(int)