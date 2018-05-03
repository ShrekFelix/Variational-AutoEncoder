import setuptools

setuptools.setup(name = "Variational_Autoencoder",
                 version = "1.0",
                 author='Weiyu Yan',
                 author_email='weiyu.yan@duke.edu',
                 url='https://github.com/ShrekFelix/Variational-AutoEncoder',
                 py_modules = ['VAE'],
                 packages=setuptools.find_packages(),
                 scripts = ['run.py'],
                 python_requires='>=3',
)
