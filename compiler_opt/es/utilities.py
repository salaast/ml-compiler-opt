"""Auxiliary functions used by RPC blackbox server and client.

Helper functions used in RPC-based distributed blackbox optimization.
"""

import csv
from absl import flags
# import matplotlib.pyplot as plt
import tensorflow as tf

FLAGS = flags.FLAGS

MARKING_FREQUENCY = 1


def write_to_csv_file(csv_file, arrays_to_write):
  with tf.gfile.Open(csv_file, 'w') as csvfile:
    writer = csv.writer(
        csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for k in range(len(arrays_to_write[0])):
      list_to_be_written = []
      for l in range(len(arrays_to_write)):
        list_to_be_written += [arrays_to_write[l][k]]
      writer.writerow(list_to_be_written)

# TODO(kchoro): Too general exception is caught, change excpetion class.


def read_from_csv_file(csv_file, array_to_write):
  try:
    with tf.gfile.Open(csv_file, 'r') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter='\t')
      for row in csv_reader:
        array_to_write.append(float(row[0]))
    return True
  except Exception as e:
    print('Error: %s' % e)
    return False


# def draw_plots(png_file, first_array, second_array, labels):
#   """Draws a plot of two curves with corresponding labels.

#   Draws a plot of two curves with corresponding labels and saves it to
#   <png_file>.

#   Args:
#     png_file: name of the png file to put the curves in
#     first_array: array encoding first curve
#     second_array: array encoding second curve
#     labels: corresponding labels
#   """
#   marking_frequency = MARKING_FREQUENCY
#   with tf.gfile.Open(png_file, 'w') as f:
#     plt.figure(1)
#     first_array_, = plt.plot(first_array[0:], markevery=marking_frequency)
#     second_array_, = plt.plot(second_array[0:], markevery=marking_frequency)
#     leg = plt.legend([first_array_, second_array_], labels)
#     leg.get_frame().set_alpha(0.5)
#     plt.savefig(f)
#     plt.close()