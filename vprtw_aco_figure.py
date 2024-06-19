import matplotlib.pyplot as plt
from multiprocessing import Queue as MPQueue

class VrptwAcoFigure:
    def __init__(self, nodes: list, path_queue: MPQueue):
        """
        Matplotlib drawing must be done in the main thread. It is recommended to create a separate thread for path finding.
        When the path finding thread finds a new path, it puts the path into the path_queue.
        The drawing thread will automatically draw it.
        Paths in the queue are stored as instances of PathMessage (class).
        Nodes in the queue are stored as instances of Node (class), using Node.x and Node.y to get the coordinates of the nodes.

        :param nodes: nodes is a list of nodes, including the depot
        :param path_queue: queue used to store paths calculated by the worker thread, each element in the queue is a path,
                           and the path contains the ids of the nodes
        """

        self.nodes = nodes
        self.figure = plt.figure(figsize=(10, 10))
        self.figure_ax = self.figure.add_subplot(1, 1, 1)
        self.path_queue = path_queue
        self._depot_color = 'k'
        self._customer_color = 'steelblue'
        self._line_color = 'darksalmon'

    def _draw_point(self):
        # Gambar depot
        self.figure_ax.scatter([self.nodes[0].x], [self.nodes[0].y], c=self._depot_color, label='depot', s=40)

        # Gambar customers
        self.figure_ax.scatter([node.x for node in self.nodes[1:]], [node.y for node in self.nodes[1:]], 
                               c=self._customer_color, label='customer', s=20)
        plt.pause(0.5)

    def run(self):
        # Pertama gambar semua node
        self._draw_point()
        self.figure.show()

        # Baca jalur baru dari antrian dan gambarkan jalur tersebut
        while True:
            if not self.path_queue.empty():
                # Get the latest path from the queue, discard other paths
                info = self.path_queue.get()
                while not self.path_queue.empty():
                    info = self.path_queue.get()

                path, distance, used_vehicle_num = info.get_path_info()
                if path is None:
                    print('[draw figure]: exit')
                    break

                # Clear current lines
                self.figure_ax.cla()
                self._draw_point()

                # Redraw lines
                self.figure_ax.set_title(f'travel distance: {distance:.2f}, number of vehicles: {used_vehicle_num}')
                self._draw_line(path)
            plt.pause(1)

    def _draw_line(self, path):
        # Draw the path according to the indices in path
        for i in range(1, len(path)):
            x_list = [self.nodes[path[i - 1]].x, self.nodes[path[i]].x]
            y_list = [self.nodes[path[i - 1]].y, self.nodes[path[i]].y]
            self.figure_ax.plot(x_list, y_list, color=self._line_color, linewidth=1.5, label='line')
            plt.pause(0.2)