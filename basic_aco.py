import numpy as np
import random
from vprtw_aco_figure import VrptwAcoFigure
from vrptw_base import VrptwGraph, PathMessage
from ant import Ant
from threading import Thread
from queue import Queue
import time

class BasicACO:
    def __init__(self, graph: VrptwGraph, ants_num=10, beta=2, tau=0.1, 
                 whether_or_not_to_show_figure=True, max_time=600):
        super()
        # graph Lokasi node dan informasi waktu layanan
        self.graph = graph
        # ants_num Jumlah semut
        self.ants_num = ants_num
        # vehicle_capacity Menunjukkan beban maksimum setiap kendaraan
        self.max_load = graph.vehicle_capacity
        # beta Pentingnya informasi heuristik
        self.beta = beta
        self.tau = tau
        # best path
        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None

        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure
        self.max_time = max_time

    def run_basic_aco(self):
        # Mulai thread untuk menjalankan basic_aco dan gunakan thread utama untuk menggambar
        path_queue_for_figure = Queue()
        basic_aco_thread = Thread(target=self._basic_aco, args=(path_queue_for_figure,))
        basic_aco_thread.start()

        # Apakah akan menampilkan gambar
        if self.whether_or_not_to_show_figure:
            figure = VrptwAcoFigure(self.graph.nodes, path_queue_for_figure)
            figure.run()
        basic_aco_thread.join()

        # Berikan None sebagai bendera akhir
        if self.whether_or_not_to_show_figure:
            path_queue_for_figure.put(PathMessage(None, None))

    def _basic_aco(self, path_queue_for_figure: Queue):
        """
        Algoritma koloni semut paling dasar
        :return:
        """
        start_time_total = time.time()

        while time.time() - start_time_total < self.max_time:
            # Atur beban kendaraan saat ini, jarak perjalanan saat ini, dan waktu saat ini untuk setiap semut
            ants = [Ant(self.graph) for _ in range(self.ants_num)]
            for ant in ants:
                # Semut perlu mengunjungi semua pelanggan
                while not ant.index_to_visit_empty():
                    next_index = self.select_next_index(ant)
                    # Tentukan apakah kondisi kendala masih terpenuhi setelah menambahkan posisinya. Jika tidak, pilih lagi dan buat penilaian lagi.
                    if not ant.check_condition(next_index):
                        next_index = self.select_next_index(ant)
                        if not ant.check_condition(next_index):
                            next_index = 0

                    # Perbarui jalur semut
                    ant.move_to_next_index(next_index)
                    self.graph.local_update_pheromone(ant.current_index, next_index)

                # Akhirnya kembali ke posisi 0
                ant.move_to_next_index(0)
                self.graph.local_update_pheromone(ant.current_index, 0)

            # Hitung panjang jalur semua semut
            paths_distance = np.array([ant.total_travel_distance for ant in ants])

            # Catat jalur terbaik saat ini
            best_index = np.argmin(paths_distance)
            if self.best_path is None or paths_distance[best_index] < self.best_path_distance:
                self.best_path = ants[int(best_index)].travel_path
                self.best_path_distance = paths_distance[best_index]
                self.best_vehicle_num = self.best_path.count(0) - 1

                # Tampilan grafis
                if self.whether_or_not_to_show_figure:
                    path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))

                print('\n')
                print('find a new path, its distance is %.0f' % (self.best_path_distance))
                print('it takes %0.2f second aco running' % (time.time() - start_time_total))

            # Perbarui tabel feromon
            self.graph.global_update_pheromone([self.best_path], paths_distance, self.best_path_distance)

        print('\n')
        print('final best path distance is %.0f, number of vehicle is %d' % (self.best_path_distance, self.best_vehicle_num))
        print('it takes %0.2f second aco running' % (time.time() - start_time_total))
        print('best path found is {}'.format(self.best_path))

        # Hitung konsumsi bahan bakar dan emisi karbon
        faktor_emisi = 0.147  # dalam kg CO2e per liter bahan bakar diesel
        emisi_karbon = self.hitung_emisi_karbon(self.best_path_distance, faktor_emisi)

        # Tampilkan hasil
        print('Emisi Karbon: {:.2f} kg CO2e'.format(emisi_karbon))

    def select_next_index(self, ant):
        """
        Pilih simpul berikutnya
        :param ant:
        :return:
        """
        current_index = ant.current_index
        index_to_visit = ant.index_to_visit

        transition_prob = self.graph.pheromone_mat[current_index][index_to_visit] * \
            np.power(self.graph.heuristic_info_mat[current_index][index_to_visit], self.beta)
        transition_prob = transition_prob / np.sum(transition_prob)

        if np.random.rand() < self.tau:
            max_prob_index = np.argmax(transition_prob)
            next_index = index_to_visit[max_prob_index]
        else:
            # Gunakan algoritma roulette
            next_index = BasicACO.stochastic_accept(index_to_visit, transition_prob)
        return next_index

    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob):
        """
        Rolet
        :param index_to_visit: a list of N index (list or tuple)
        :param transition_prob:
        :return: selected index
        """
        # calculate N and max fitness value
        N = len(index_to_visit)

        # normalize
        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob / sum_tran_prob

        # select: O(1)
        while True:
            # randomly select an individual with uniform probability
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]

    def hitung_emisi_karbon(self, jarak_tempuh, faktor_emisi):
        return jarak_tempuh * faktor_emisi