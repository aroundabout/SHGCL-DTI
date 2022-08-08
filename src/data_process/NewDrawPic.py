import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


class PicDrawer():
    def __init__(self, dir='pic/sens.svg', x_size=18, y_size=9):
        self.x_size = x_size
        self.y_size = y_size
        self.dic = dir
        self.number = 0

    def _getColor(self, bar_number: int) -> list:
        x = np.arange(bar_number)
        y = x ** 2
        norm = plt.Normalize(y.min(), y.max())
        norm_y = norm(y)
        map_vir = cm.get_cmap(name='viridis')
        color = map_vir(norm_y)
        return color

    def _getColorTwo(self):
        return ['black', 'red']

    def _getLabel(self):
        return ['AUROC', 'AUPR']

    def _getAxs(self, row, col, dtype):
        axs = [plt.subplot(row, col, i + 1) for i in range(row * col)]
        if dtype == 'double':
            grid = plt.GridSpec(2, 3, wspace=0.5, hspace=0.5)
            axs = [plt.subplot(grid[0, 0]),
                   plt.subplot(grid[0, 1:3]),
                   plt.subplot(grid[1, 0]),
                   plt.subplot(grid[1, 1:3])]
        return axs

    def drawMult(self, row, col, x_data_list, y_data_list, low_list, high_list, std_err_list, title_list, dtype,
                 isNumber):
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(self.x_size, self.y_size))
        plt.subplots_adjust(wspace=0.2, hspace=0.5)
        drawFunc = self._drawBars
        if dtype == 'bar':
            drawFunc = self._drawBars
        elif dtype == 'errorbar':
            drawFunc = self._drawErrorBars
        elif dtype == 'double':
            drawFunc = self._drawDoubleBar
        axs = self._getAxs(row, col, dtype)
        for i in range(row * col):
            if std_err_list is None or len(std_err_list) == 0:
                drawFunc(axs[i], x_data_list[i], y_data_list[i], low_list[i], high_list[i], title_list[i], isNumber)
            else:
                drawFunc(axs[i], x_data_list[i], y_data_list[i], low_list[i], high_list[i], title_list[i], isNumber,
                         std_err_list[i])
        plt.savefig(self.dic, format='svg')
        plt.show()
        self.number = 0

    def _drawBars(self, ax, x_data: list, y_data: list, low, high, subtitle, isNumber=True, sd=None):
        color = self._getColor(len(x_data))
        plt.sca(ax)
        ax.set_title(subtitle)
        if isNumber:
            plt.text(0, 1.25, '(' + chr(97 + self.number) + ')', transform=ax.transAxes,
                     fontdict={'family': 'SimHei', 'size': 24, 'weight': 'bold'})
            self.number += 1
        error_params = dict(elinewidth=4, ecolor='grey', capsize=5)
        plt.bar(x_data, y_data, 0.5, color=color, edgecolor='grey', yerr=sd, error_kw=error_params)
        plt.ylim(low, high)
        if sd is None or len(sd) == 0:
            for a, b in zip(x_data, y_data):
                plt.text(a, b, '%.4f' % b, ha='center', va='bottom')

    def _drawErrorBars(self, ax, x_data: list, y_data: list, low, high, subtitle, isNumber=True, sd_list=None):
        plt.sca(ax)
        ax.set_title(subtitle)
        if isNumber:
            plt.text(0, 1.03, '(' + chr(97 + self.number) + ')', transform=ax.transAxes,
                     fontdict={'family': 'SimHei', 'size': 24, 'weight': 'bold'})
            self.number += 1
        colors = self._getColorTwo()
        labels = self._getLabel()
        for y, sd, color, label in zip(y_data, sd_list, colors, labels):
            plt.errorbar(x_data, y, yerr=sd, elinewidth=2, capsize=4, ecolor=color, color=color, label=label)
        plt.ylim(low, high)
        plt.legend()
        if sd_list is None or len(sd_list) == 0:
            for a, b in zip(x_data, y_data):
                plt.text(a, b, '%.4f' % b, ha='center', va='bottom')

    def _drawDoubleBar(self, ax, x_data: list, y_data: list, low, high, subtitle, isNumber, sd_list=None):
        color = self._getColor(6)
        plt.sca(ax)
        bar_width = 0.25
        ax.set_title(subtitle)
        ax.autoscale(enable=True)
        if isNumber:
            plt.text(0, 1.25, '(' + chr(97 + self.number) + ')', transform=ax.transAxes,
                     fontdict={'family': 'SimHei', 'size': 24, 'weight': 'bold'})
            self.number += 1
        pos = np.arange(len(y_data[0]))
        error_params = dict(elinewidth=4, ecolor='grey', capsize=5)
        plt.bar(pos, y_data[0], bar_width, color=[color[3]], label='AUROC', yerr=sd_list[0], edgecolor='grey',
                error_kw=error_params)
        plt.bar(pos + bar_width, y_data[1], bar_width, color=[color[4]], label='AUPR', yerr=sd_list[1],
                edgecolor='grey',
                error_kw=error_params)
        plt.legend()
        plt.xticks(pos + bar_width / 2, x_data)
        plt.ylim(low, high)
        return 0


def getSensitivityLayers():
    row = 1
    col = 1
    x_data_list = [['1', '2', '3']]
    y_data_list = [[[0.9550, 0.9582, 0.9548],
                    [0.8765, 0.8838, 0.8775]]]
    low_list = [0.8]
    high_list = [1]
    std_err = [[[0.0015, 0.0016, 0.0017],
                [0.0018, 0.0023, 0.0030]]]
    title_list = ['Different choices of GCN layers']
    return row, col, x_data_list, y_data_list, low_list, high_list, std_err, title_list, 'errorbar', False


def getSensitivityDims():
    row = 1
    col = 1
    x_data_list = [['512', '768', '1024', '1536', '2048', '3072']]
    y_data_list = [[[0.9562, 0.9574, 0.9581, 0.9573, 0.9582, 0.9586],
                    [0.8727, 0.8777, 0.8798, 0.8814, 0.8838, 0.8821]]]
    low_list = [0.8]
    high_list = [1]
    std_err = [[[0.0011, 0.0013, 0.0011, 0.0013, 0.0016, 0.0009],
                [0.0013, 0.0025, 0.0019, 0.0019, 0.0023, 0.0018]]]
    title_list = ['Different choices of hidden dims']
    return row, col, x_data_list, y_data_list, low_list, high_list, std_err, title_list, 'errorbar', False


def getSensitivity():
    row = 1
    col = 2
    x_data_list = [['1', '2', '3'], ['512', '768', '1024', '1536', '2048', '3072']]
    y_data_list = [[[0.9550, 0.9582, 0.9548],
                    [0.8765, 0.8838, 0.8775]],
                   [[0.9562, 0.9574, 0.9581, 0.9573, 0.9582, 0.9586],
                    [0.8727, 0.8777, 0.8798, 0.8814, 0.8838, 0.8821]]]
    low_list = [0.8, 0.8]
    high_list = [1, 1]
    std_err = [[[0.0015, 0.0016, 0.0017],
                [0.0018, 0.0023, 0.0030]],
               [[0.0011, 0.0013, 0.0011, 0.0013, 0.0016, 0.0009],
                [0.0013, 0.0025, 0.0019, 0.0019, 0.0023, 0.0018]]]
    title_list = ['Different choices of GCN layers', 'Different choices of hidden dims']
    return row, col, x_data_list, y_data_list, low_list, high_list, std_err, title_list, 'errorbar', True


def getCompareAuc():
    row = 2
    col = 3
    x_data_list = [['NetLapRLS', 'DTINet', 'NeoDTI', 'EEG-DTI', 'SHGCL-DTI'] * 1] * 6
    y_data_list = [[0.9057, 0.9158, 0.9458, 0.9517, 0.9582],
                   [0.8893, 0.9070, 0.9147, 0.9524, 0.9232],
                   [0.8706, 0.8408, 0.8912, 0.8820, 0.9098],
                   [0.8978, 0.9063, 0.9329, 0.9368, 0.9483],
                   [0.8968, 0.9129, 0.9441, 0.9476, 0.9532],
                   [0.7485, 0.6752, 0.7373, 0.7045, 0.7615]]
    low_list = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    high_list = [1, 1, 1, 1, 1, 1]
    std_err = [[0.0080, 0.0019, 0.0018, 0.0011, 0.0016],
               [0.0016, 0.0020, 0.0039, 0.0008, 0.0024],
               [0.0041, 0.0035, 0.0032, 0.0061, 0.0023],
               [0.0101, 0.0024, 0.0038, 0.0015, 0.0016],
               [0.0050, 0.0015, 0.0021, 0.0016, 0.0019],
               [0.0328, 0.0026, 0.0093, 0.0061, 0.0081]]
    title_list = ['AUROC\npositive: negative=1:10\n \n ',
                  'AUROC\nall unknown DPP were\ntreated as negative samples\n ',
                  'AUROC\npositive: negative=1:10\nDTIs with similar drugs\nor proteins were removed',
                  'AUROC\npositive: negative=1:10\nDTIs with drugs sharing similar\ndrug interactions were removed',
                  'AUROC\npositive: negative=1:10\nDTIs with drugs sharing similar\nside-effect were removed',
                  'AUROC\ntrained on non-unique DTIs\ntested on unique DTIs\n ']
    return row, col, x_data_list, y_data_list, low_list, high_list, std_err, title_list, 'bar', True


def getCompareAupr():
    row = 2
    col = 3
    x_data_list = [['NetLapRLS', 'DTINet', 'NeoDTI', 'EEG-DTI', 'SHGCL-DTI'] * 1] * 6
    y_data_list = [[0.7474, 0.8183, 0.8528, 0.8442, 0.8838],
                   [0.4077, 0.4402, 0.5806, 0.1698, 0.6075],
                   [0.5967, 0.6099, 0.6868, 0.6753, 0.7382],
                   [0.7196, 0.7573, 0.8171, 0.8057, 0.8525],
                   [0.7233, 0.8090, 0.8438, 0.8402, 0.8764],
                   [0.2726, 0.3778, 0.4304, 0.1867, 0.4505]]
    low_list = [0.4, 0, 0.4, 0.4, 0.4, 0]
    high_list = [0.9, 0.7, 0.8, 0.9, 0.9, 0.6]
    std_err = [[0.0140, 0.0026, 0.0042, 0.0021, 0.0023],
               [0.0060, 0.0037, 0.0044, 0.0091, 0.0029],
               [0.0085, 0.0117, 0.0097, 0.0083, 0.0058],
               [0.0172, 0.0029, 0.0069, 0.0021, 0.0037],
               [0.0140, 0.0023, 0.0078, 0.0032, 0.0021],
               [0.0202, 0.0057, 0.0157, 0.0047, 0.0164]]
    title_list = ['AUPR\npositive: negative=1:10\n \n ',
                  'AUPR\nall unknown DPP were\ntreated as negative samples\n ',
                  'AUPR\npositive: negative=1:10\nDTIs with similar drugs\nor proteins were removed',
                  'AUPR\npositive: negative=1:10\nDTIs with drugs sharing similar\ndrug interactions were removed',
                  'AUPR\npositive: negative=1:10\nDTIs with drugs sharing similar\nside-effect were removed',
                  'AUPR\ntrained on non-unique DTIs\ntested on unique DTIs\n ']
    return row, col, x_data_list, y_data_list, low_list, high_list, std_err, title_list, 'bar', True


def getRoubest():
    row = 2
    col = 2
    x_data_list = [['all network', 'Remove side-effect\nassociation network'],
                   ['all network', 'remove DDI network', 'remo PPI network', 'remove both'],
                   ['all network', 'Remove disease\nassociation network'],
                   ['all network', 'remove drug\nsimilarity network', 'remove protein\nsimilarity network',
                    'remove both']]
    y_data_list = [[[0.9582, 0.9564], [0.8838, 0.8802]],
                   [[0.9582, 0.9580, 0.9537, 0.9534], [0.8838, 0.8822, 0.8684, 0.8677]],
                   [[0.9582, 0.9442], [0.8838, 0.8580]],
                   [[0.9582, 0.9572, 0.9588, 0.9577], [0.8838, 0.8817, 0.8817, 0.8799]]]
    low_list = [0.8, 0.8, 0.8, 0.8]
    high_list = [1, 1, 1, 1]
    std_err = [[[0.0016, 0.0018, ], [0.0023, 0.0019]],
               [[0.0016, 0.0010, 0.0014, 0.0011], [0.0023, 0.0013, 0.0017, 0.0017]],
               [[0.0016, 0.0023], [0.0019, 0.0018]],
               [[0.0016, 0.0011, 0.0014, 0.0006], [0.0023, 0.0013, 0.0023, 0.0017]]]
    title_list = ['Remove side-effect\nassociation network',
                  'Remove drug and\nprotein interaction networks',
                  'Remove disease\nassociation network',
                  'Remove drug and\nprotein similarity networks']
    return row, col, x_data_list, y_data_list, low_list, high_list, std_err, title_list, 'double', True


def getDefault(func=None):
    if func is not None:
        return func()
    row = 2
    col = 1
    x_data_list = [['test1', 'niu2', 'buniu3', 'exp4'], ['test1', 'niu2', 'buniu3', 'exp4']]
    y_data_list = [[0.4, 0.3, 0.1, 0.2], [0.1, 0.2, 0.3, 0.9]]
    low_list = [0, 0, 0, 0]
    high_list = [1, 1, 1, 1]
    title_list = ['1', '1']
    return row, col, x_data_list, y_data_list, low_list, high_list, None, title_list, 'bar'


if __name__ == '__main__':
    r, c, x_list, y_list, l_list, h_list, e_list, t_list, dtype, isN = getDefault(getCompareAupr)
    drawer = PicDrawer(x_size=18, y_size=9,dir='pic/compareAupr.svg')
    drawer.drawMult(r, c, x_list, y_list, l_list, h_list, e_list, t_list, dtype, isN)
