#python3
#we are using google_images_download library
import os
from google_images_download import google_images_download   #importing the library


def google_image_download(file_path = './image_data/'):
    '''this function helps to download images based on user inputs'''
    
    #class instantiation
    response = google_images_download.googleimagesdownload() 
    image_keywords = input('please input image keywords to download images from google, separated by ",": ')
    num_of_image = int(input('please input the number of images that you would like to download: '))
    image_size = input('please input the size of the images to be downloaded: ')
    #you need to install chromedriver and selenium to be able to download more than 100 images at a time
    # for mac users, please install chromedriver first by typing: brew cask install chromedriver,
    #and then specify the path to your chromedriver indicated by brew after installation
    arguments = {'keywords': image_keywords, 'output_directory': file_path, 'size': image_size,
                'limit': num_of_image, 'chromedriver': '/usr/local/bin/chromedriver'}
    
    return response.download(arguments)

if __name__ == '__main__':
    google_image_download()





