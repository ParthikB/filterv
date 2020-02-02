import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


class Filter:
	def __init__(self):
		self.image          = None
		self.original_img 	= None
		self.output_img 	= None
		self.FILTER_SIZE 	= None
		self.STRIDE 		= None
		self.filter_vals 	= None
		self.img_row 		= None
		self.img_col 		= None
		self.output_row_dim = None
		self.output_col_dim = None

	def load_img(self, path, resize=None):
		img = cv2.imread(path, 0)
		original_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
		if resize:
			img = cv2.resize(img, (resize, resize)) # Resizing temporarily for faster calculations

		self.image        = img
		return original_img


	def output_dim(self, input_size, filter_size, stride):
		dim = (input_size-filter_size)/stride + 1
		
		return int(dim)


	def __input_filter_val(self, f):
		try:
			val = int(input(f'f{f+1} : '))
		except:
			print('Invalid Val. Try Again..')
			val = self.__input_filter_val(f)
		return val


	def parameters(self, STRIDE=0, FILTER_SIZE=0, filter_vals=[]):
		if not STRIDE: 		STRIDE			 = int(input('Enter Stride      : '))
		if not FILTER_SIZE: FILTER_SIZE 	 = int(input('Enter Filter Size : '))

		# filter_vals		 = [1, 1, 1, 0, 0, 0, -1, -1, -1]
		if not filter_vals:
			for i in range(FILTER_SIZE**2):
				val = self.__input_filter_val(i)
				filter_vals.append(val)

		elif len(filter_vals) != FILTER_SIZE**2:
				raise ValueError('Invalid number of Filter Values.')

		
		img_row, img_col = self.image.shape
		output_row_dim 	 = self.output_dim(img_row, FILTER_SIZE, STRIDE)
		output_col_dim 	 = self.output_dim(img_col, FILTER_SIZE, STRIDE)

		self.FILTER_SIZE 	= FILTER_SIZE
		self.STRIDE 		= STRIDE
		self.filter_vals 	= filter_vals
		self.img_row 		= img_row
		self.img_col 		= img_col
		self.output_row_dim = output_row_dim
		self.output_col_dim = output_col_dim



	def convert(self):
		output = []
		# Try doing this in one for-loop by unrolling the image to a vector.
		for i in tqdm(range(0, self.img_row-self.FILTER_SIZE+1, self.STRIDE)):
			rows = [i+x for x in range(self.FILTER_SIZE)]

			for j in range(0, self.img_col-self.FILTER_SIZE+1, self.STRIDE):
				cols = [j+x for x in range(self.FILTER_SIZE)]

				img_vector = []
				for row in rows:
					for col in cols:
						pixel = self.image[row][col] 
						img_vector.append(pixel)

				output_pixel = np.sum(np.multiply(img_vector, self.filter_vals))
				output.append(output_pixel)


		output = np.array(output)
		output_img = output.reshape(self.output_row_dim, self.output_col_dim)
		self.output_img = output_img

		return output_img
		

	def plot(self):
		ax1 = plt.subplot(1, 2, 1)
		ax1.imshow(self.original_img)
		plt.title('Original Image')
		ax2 = plt.subplot(1, 2, 2)
		ax2.imshow(self.output_img, cmap='gray')
		plt.title('Filtered Image')

		plt.show()


if __name__ == '__main__':
	
	f = Filter()

	img = f.load_img('dummy.jpg')

	f.parameters()

	output_img = f.convert()

	f.plot(img, output_img)
