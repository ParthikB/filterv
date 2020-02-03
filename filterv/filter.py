import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


class Filter:

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


	def parameters(self, stride: int, filter_size: int, filter_vals: list=[]):

		if type(stride) != int or type(filter_size) != int:
				raise ValueError('Invalid parameter type.')
		
		# filter_vals		 = [1, 1, 1, 0, 0, 0, -1, -1, -1]
		if not filter_vals:
			for i in range(filter_size**2):
				val = self.__input_filter_val(i)
				filter_vals.append(val)

		elif len(filter_vals) != filter_size**2:
				raise ValueError('Invalid number of Filter Values.')

		
		
		img_row, img_col = self.image.shape
		output_row_dim 	 = self.output_dim(img_row, filter_size, stride)
		output_col_dim 	 = self.output_dim(img_col, filter_size, stride)

		self.filter_size 	= filter_size
		self.stride 		= stride
		self.filter_vals 	= filter_vals
		self.img_row 		= img_row
		self.img_col 		= img_col
		self.output_row_dim = output_row_dim
		self.output_col_dim = output_col_dim



	def convert(self):
		output = []
		# Try doing this in one for-loop by unrolling the image to a vector.
		for i in tqdm(range(0, self.img_row-self.filter_size+1, self.stride)):
			rows = [i+x for x in range(self.filter_size)]

			for j in range(0, self.img_col-self.filter_size+1, self.stride):
				cols = [j+x for x in range(self.filter_size)]

				img_vector = []
				for row in rows:
					for col in cols:
						pixel = self.image[row][col] 
						img_vector.append(pixel)

				output_pixel = np.sum(np.multiply(img_vector, self.filter_vals))
				output.append(output_pixel)


		output = np.array(output)
		output_img = output.reshape(self.output_row_dim, self.output_col_dim)

		return output_img
		

	def compare(self, original_img, filtered_img):
		ax1 = plt.subplot(1, 2, 1)
		ax1.imshow(original_img)
		plt.title('Original Image')
		ax2 = plt.subplot(1, 2, 2)
		ax2.imshow(filtered_img, cmap='gray')
		plt.title('Filtered Image')

		plt.show()

		

if __name__ == '__main__':
	
	# Initializing the Filter Class
	f = Filter()
	
	# Loading the image.
	original_img = f.load_img(path='dummy.jpg')
	
	# Parameter dive
	f.parameters(stride=2, 
		     filter_size=3, 
		     filter_vals=[1, 1, 1, 0, 0, 0, -1, -1, -1])
	
	# Converting
	filtered_img = f.convert()
	
	# Comparing the two images
	f.compare(original_img=original_img, 
	          filtered_img=filtered_img)
