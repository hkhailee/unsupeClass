"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        path = '/bsuhome/hkiesecker/scratch/imageClassification/US/'
        #alt_path = '/bsuhome/hkiesecker/scratch/imageClassification/UI_ss/'
        db_names = {'cifar-10', 'stl-10', 'cifar-20', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200','rico_20'}
        assert(database in db_names)

        if database =='rico_20':
            return path+'/rico_20/'

        if database == 'cifar-10':
            return path+'/cifar-10/'
        
        elif database == 'cifar-20':
            return path+'/cifar-20/'

        elif database == 'stl-10':
            return path+'/stl-10/'
        
        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return path+'/imagenet/'
        
        else:
            raise NotImplementedError
