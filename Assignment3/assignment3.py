import os, glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


"""
Name: Huerta, Emilia (Please write names in <Last Name, First Name> format)

Collaborators: Kitamura, Masao (Please write names in <Last Name, First Name> format)

Collaboration details: Discussed <function name> implementation details with Jane Doe.
"""

CELEBA_DIRPATH = 'celebA_dataset'
N_HEIGHT = 78
N_WIDTH = 78
N_TRAIN = 850

def get_eigenfaces(eigenvalues, eigenvectors, k):
  """
    Sorts the eigenvector by eigenvalues
    Returns the projection matrix (eigenfaces)

    faces_centered : N x d vector
      mean subtracted faces
    eigenvalues : 1 x d vector
      eigenvalues
    eigenvectors : d x d vector
      eigenvectors
    k : int
      number of eigenfaces to produce

    returns d x k vector
  """
  #sort eigenvalues

  #take order

  #sort eigenvectors

  #select from o to k

  #eigenvectors

def project(faces, faces_mean, eigenfaces):
  """
    Returns the projected faces (lower dimensionality)

    faces : N x d vector
      images of faces
    faces_mean : 1 x d vector
      per pixel average of images of faces
    eigenfaces : d x k vector
      projection matrix

    returns N x k vector
  """
  #Slide #28 - Compute the mean & center the data (faces)
  #B = x - mean of x
  B = faces - faces_mean

  #Multiply centered faces with eigenfaces
  return np.matmul(B, eigenfaces)  #(B_train, W)

def reconstruct(faces_projected, faces_mean, eigenfaces):
  """
    Returns the reconstructed faces (back-projected)

    faces_projected : N x k vector
      faces in lower dimensions
    faces_mean : 1 x d vector
      per pixel average of images of faces
    eigenfaces : d x k vector
      projection matrix

    returns N x d vector
  """
  #Slide #29 - Recover oour data
  #X_train_hat = np.matmul(Z_train, W.T) + mu_train
  return np.matmul(faces_projected, eigenfaces.T) + faces_mean

def synthesize(eigenfaces, variances, faces_mean, k=50, n=25):
  """
    Synthesizes new faces by sampling from the latent vector distribution

    eigenfaces : d x k vector
      projection matrix
    variances : 1 x d vector
      variances
    faces_mean : 1 x d vector

    returns synthesized faces
  """
  #sample from distribution of Z
  np.random(0, np.sqrt(sigma))


def mean_squared_error(x, x_hat):
  """
    Computes the mean squared error

    x : N x d vector
    x_hat : N x d vector

    returns mean squared error
  """
  return np.mean((x - x_hat) ** 2) #correct


def plot_eigenvalues(eigenvalues):
  """
    Plots the eigenvalues from largest to smallest

    eigenvalues : 1 x d vector
  """
  fig = plt.figure()
  fig.suptitle('Eigenvalues Versus Principle Components')


def visualize_reconstructions(faces, faces_hat, n=4):
  """
    Creates a plot of 2 rows by n columns
    Top row should show original faces
    Bottom row should show reconstructed faces (faces_hat)

    faces : N x k vector
      images of faces
    faces_hat : 1 x d vector
      images reconstructed faces
  """
  fig = plt.figure()
  fig.suptitle('Real Versus Reconstructed Faces')


def plot_reconstruction_error(mses, k):
  """
    Plots the reconstruction errors

    mses : list
      list of mean squared errors
    ks : list
      list of k used
  """
  fig = plt.figure()
  fig.suptitle('Reconstruction Error')


def visualize_eigenfaces(eigenfaces):
  """
    Creates a plot of 5 rows by 5 columns
    Shows the first 25 eigenfaces (principal components)
    For each dimension k, plot the d number values as an image

    eigenfaces : d x k vector
  """
  #Slide 34 - Visualizing the data
  #faces_train = np.reshape(faces_train, (-1, 78, 78))
  fig = plt.figure()
  fig.suptitle('Top 25 Eigenfaces')
  for i in range(0, 25):
    ax = fig.add_subplot(5, 5, i + 1)
    ax.imshow(faces_train[i, ...], cmap='gray')
  plt.show()

  # fig = plt.figure()
  # for i in range(5 * 5): #25
  #   ax = fig.add_subplot(5, 5, i + 1)
  #   plt.imshow(faces[i, ...], cmap='gray')



    # if i < n:
    #   plt.imshow(faces[i, ...],cmap = 'gray')
    # else:
    #   plt.imshow(reconstructed_faces[i-n, ...], cmap='gray')


  #fig.suptitle('Top 25 Eigenfaces')


def visualize_synthetic_faces(faces):
  """
    Creates a plot of 5 rows by 5 columns
    Shows the first 25 synthetic faces

    eigenfaces : N x d vector
  """
  #X = ZW^T + U
  #Z = BW projects B to subspace    -------this should give you the projection  np.matmal(B,W)
            #B is training data minus the mean of your training set (centers your data)
            #Passed in a B_train
  #W is eigenfaces which projects
  #wT will project back to original space

  fig = plt.figure()
  fig.suptitle('Synthetic Faces')


if __name__ == '__main__':
  # Load faces from directory
  face_image_paths = glob.glob(os.path.join(CELEBA_DIRPATH, '*.jpg'))

  print('Loading {} images from {}'.format(len(face_image_paths), CELEBA_DIRPATH))
  # Read images as grayscale and resize from (128, 128) to (78, 78)
  faces = []
  for path in face_image_paths:
    im = Image.open(path).convert('L').resize((N_WIDTH, N_HEIGHT))
    faces.append(np.asarray(im))
  faces = np.asarray(faces) # (1000, 78, 78)
  # Normalize faces between 0 and 1
  faces = faces/255.0   #faces is X

  print('Vectorizing faces into N x d matrix')
  # TODO: reshape the faces to into an N x d matrix
  #Slide 26 - reshape the image into a vector
  faces = np.reshape(faces, (-1, 6084))  # 78 * 78 = 6084

  print('Splitting dataset into {} for training and {} for testing'.format(N_TRAIN, faces.shape[0]-N_TRAIN))
  faces_train = faces[0:N_TRAIN, ...]
  faces_test = faces[N_TRAIN::, ...]

  X_train = faces_train
  X_test = faces_test

  print('Computing eigenfaces from training set')
  # TODO: obtain eigenfaces and eigenvalues
  #Slide 28 - compute the mean ans center the data
  mu_train = np.mean(faces_train, axis=0)
  B_train = faces_train - mu_train

  #Compute the covariance matrix
  C = np.matmul(B_train.T, B_train) / (B_train.shape[0])

  # Eigen decomposition
  S, V = np.linalg.eig(C)

  print('Plotting the eigenvalues from largest to smallest')
  # TODO: plot the first 200 eigenvalues from largest to smallest
  # Sort the values in descending order
  #NOT SURE IF THIS WORKS
  top_values = np.sort(S)
  top_values = np.flip(top_values)

  # Select the top 45 dimensions
  #Slide #31
  top_values = V[0:200]

  plt.title("Top 200 Eigenvalues")
  plt.xlabel('Ranking')
  plt.ylabel('Eigenvalues')
  plt.plot(top_values)
  plt.show()  #takes too long

  print('Visualizing the top 25 eigenfaces')
  # TODO: visualize the top 25 eigenfaces
  # #Slide 34 - Visualizing the data
  faces_train = np.reshape(faces_train, (-1, 78, 78))
  fig = plt.figure()
  fig.suptitle('Top 25 Eigenfaces')
  for i in range(0, 25):
    ax = fig.add_subplot(5, 5, i + 1)
    ax.imshow(faces_train[i, ...], cmap='gray')
  plt.show()

  # fig = plt.figure()
  # for i in range(5 * 5): #25
  #   ax = fig.add_subplot(5, 5, i + 1)
  #   plt.imshow(faces[i, ...], cmap='gray')

  #visualize_eigenfaces()

  print('Plotting training reconstruction error for various k')
  # TODO: plot the mean squared error of training set with
  # k=[5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200]

  print('Reconstructing faces from projected faces in training set')
  # TODO: choose k and reconstruct training set

  # TODO: visualize the reconstructed faces from training set

  print('Reconstructing faces from projected faces in testing set')
  # TODO: reconstruct faces from the projected faces

  # TODO: visualize the reconstructed faces from training set

  print('Plotting testing reconstruction error for various k')
  # TODO: plot the mean squared error of testing set with
  # k=[5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200]

  print('Creating synethic faces')
  # TODO: synthesize and visualize new faces based on the distribution of the latent variables
  # you may choose another k that you find suitable
