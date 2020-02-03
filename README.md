# filterV


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

*filterV is a short program that enables the user to visualize how any image would look after applying the Convolutional filters.*


### Installation

filterV requires python3 to run.

Enter the following command in your command-line (windows):
```sh
pip install filterv
```

### Using filterV

It's super easy to use...!

    import filterv.filter as f             # import the library
    F = f.Filter()                         # create a Class Instance

    input_img  = F.load_img(path_to_image) # load the image
    F.parameters()                         # Initialize the parameters
    output_img = F.convert()               # Filter the Image
    
    F.plot(input_img, output_img)          # Plot the outputs
    
License
--------

MIT


**Free Software, Hell Yeah!**

