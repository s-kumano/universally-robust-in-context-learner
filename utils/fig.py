import os
from typing import (Any, Iterator, List, Literal, Optional, Sequence, Tuple,
                    overload)

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import seaborn as sns
from matplotlib.axis import Axis
from matplotlib.cm import ScalarMappable
from matplotlib.collections import (LineCollection, PathCollection,
                                    PolyCollection)
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.container import BarContainer
from matplotlib.contour import QuadContourSet
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.pyplot import Axes as PltAxes
from matplotlib.quiver import Quiver
from matplotlib.text import Text

'''
filled_markers = ('.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
'''


class Axes:
    def __init__(self, ax: PltAxes) -> None:
        self.ax = ax

    def remove(self) -> None:
        self.ax.remove()

    def remove_grid(self) -> None:
        self.ax.grid(False)

    def square(self) -> None:
        self.ax.set_box_aspect(1)

    def set_spines_linewidth(self, w: float) -> None:
        for spine in self.ax.spines.values():
            spine.set_linewidth(w)

    def set_fontsize(self, fontsize: float) -> None:
        self.set_title_fontsize(fontsize)
        self.set_label_fontsize(fontsize)
        self.set_ticks_fontsize(fontsize)
        self.set_legend_fontsize(fontsize)

    def set_clabel(self, cs: QuadContourSet, **kwargs) -> None:
        '''
        fontsize: Optional[float] = None
            Size in points or relative size.
        inline: bool = True
            If True the underlying contour is removed where the label is placed.
        inline_spacing: float = 5.
            Space in pixels to leave on each side of label when placing inline.
        fmt: str = '%1.3f'
            A format string for the label.
        manual: Optional[Tuple[Tuple[float, float], ...]] = None
            An iterable object of x,y tuples.
        '''
        self.ax.clabel(cs, **kwargs)

    ######################## Title ########################

    def set_title(self, title: str, fontsize: Optional[float] = None) -> None:
        self.ax.set_title(title, fontsize=fontsize or plt.rcParams['axes.titlesize'])

    def set_title_fontsize(self, fontsize: float) -> None:
        self.ax.set_title(self.ax.get_title(), fontsize=fontsize)

    ######################## xy labels ########################

    def set_xlabel(
        self, 
        label: str, 
        fontsize: Optional[float] = None, 
        labelpad: Optional[float] = None,
    ) -> None:
        self.ax.set_xlabel(
            label, 
            fontsize=fontsize or plt.rcParams['axes.labelsize'], 
            labelpad=labelpad, # type: ignore
        )

    def set_ylabel(
        self, 
        label: str, 
        fontsize: Optional[float] = None, 
        labelpad: Optional[float] = None,
    ) -> None:
        self.ax.set_ylabel(
            label, 
            fontsize=fontsize or plt.rcParams['axes.labelsize'], 
            labelpad=labelpad, # type: ignore
        )

    def set_label_fontsize(self, fontsize: float) -> None:
        self.set_xlabel_fontsize(fontsize)
        self.set_ylabel_fontsize(fontsize)

    def set_xlabel_fontsize(self, fontsize: float) -> None:
        self.ax.set_xlabel(self.ax.get_xlabel(), fontsize=fontsize)

    def set_ylabel_fontsize(self, fontsize: float) -> None:
        self.ax.set_ylabel(self.ax.get_ylabel(), fontsize=fontsize)

    @staticmethod
    def _move_label(axis: Axis, left: float, down: float) -> None:
        label = axis.get_label()
        x, y = label.get_position()
        label.set_position((x-left, y-down))

    def move_xlabel(self, left: float, down: float) -> None:
        self._move_label(self.ax.xaxis, left, down)

    def move_ylabel(self, left: float, down: float) -> None:
        self._move_label(self.ax.yaxis, left, down)

    ######################## xy ticks ########################

    def set_xticks(self, ticks: Sequence[float]) -> None:
        self.ax.set_xticks(ticks)

    def set_yticks(self, ticks: Sequence[float]) -> None:
        self.ax.set_yticks(ticks)

    def set_ticks_fontsize(self, fontsize: float) -> None:
        self.ax.tick_params(axis='both', which='major', labelsize=fontsize)

    def set_xticks_fontsize(self, fontsize: float) -> None:
        self.ax.tick_params(axis='x', which='major', labelsize=fontsize)

    def set_yticks_fontsize(self, fontsize: float) -> None:
        self.ax.tick_params(axis='y', which='major', labelsize=fontsize)

    def get_xticklabels(self) -> List[Text]:
        return self.ax.get_xticklabels()

    def get_yticklabels(self) -> List[Text]:
        return self.ax.get_yticklabels()

    def set_xticklabels(self, ticks: Sequence[Any]) -> None:
        self.ax.set_xticklabels(ticks)

    def set_yticklabels(self, ticks: Sequence[Any]) -> None:
        self.ax.set_yticklabels(ticks)

    def set_xlim(self, left: float, right: float) -> None:
        self.ax.set_xlim(left, right)

    def set_ylim(self, bottom: float, top: float) -> None:
        self.ax.set_ylim(bottom, top)

    def remove_xticks(self) -> None:
        self.ax.tick_params(bottom=False, labelbottom=False)

    def remove_yticks(self) -> None:
        self.ax.tick_params(left=False, labelleft=False)

    def remove_ticks(self) -> None:
        self.remove_xticks()
        self.remove_yticks()

    def set_ticks_pad(self, pad: float) -> None:
        self.ax.tick_params(pad=pad)

    def set_xticks_pad(self, pad: float) -> None:
        self.ax.xaxis.set_tick_params(pad=pad)

    def set_yticks_pad(self, pad: float) -> None:
        self.ax.yaxis.set_tick_params(pad=pad)

    def set_xlogscale(self) -> None:
        self.ax.set_xscale('log')

    def set_ylogscale(self) -> None:
        self.ax.set_yscale('log')

    ######################## Formatter ########################
        
    def set_xticks_strformat(self, format: str) -> None:
        # ex. '{x:.1f}'
        self.ax.xaxis.set_major_formatter(tkr.StrMethodFormatter(format))

    def set_yticks_strformat(self, format: str) -> None:
        self.ax.yaxis.set_major_formatter(tkr.StrMethodFormatter(format))

    def set_xticks_offset(self, use: bool) -> None:
        self.ax.xaxis.set_major_formatter(tkr.ScalarFormatter(use))

    def set_yticks_offset(self, use: bool) -> None:
        self.ax.yaxis.set_major_formatter(tkr.ScalarFormatter(use))

    def set_xticks_comma(self) -> None:
        self.ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, _: f'{int(x):,}'))

    def set_yticks_comma(self) -> None:
        self.ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, _: f'{int(x):,}'))

    ######################## Legend ########################

    def set_legend_fontsize(self, fontsize: float) -> None:
        self.ax.legend(fontsize=fontsize)

    def legend(self, *args, **kwargs) -> Legend:
        '''
        loc: Literal['best', 'upper left'] = 'best'
            The location of the legend.
        bbox_to_anchor: Sequence[float] = axes.bbox
            Box that is used to position the legend in conjunction with loc.
        ncols: int = 1
            The number of columns that the legend has.
        frameon: bool = True
            Whether the legend should be drawn on a patch (frame).
        borderpad: float = .4
            The fractional whitespace inside the legend border, in font-size units.
        columnspacing: float = 2.
            The spacing between columns, in font-size units.
        handlelength: float = 2.
            The length of the legend handles, in font-size units.
        handleheight: float = .7
            The height of the legend handles, in font-size units.
        handletextpad: float = .8
            The pad between the legend handle and text, in font-size units.
        '''
        return self.ax.legend(*args, **kwargs)
    
    def set_legend_bottom(self, *args, offset: Tuple[float, float] = (0, 0), **kwargs) -> Legend:
        return self.ax.legend(*args, loc='upper center', bbox_to_anchor=(.5+offset[0], 0+offset[1]), **kwargs)
    
    def set_legend_right(self, *args, offset: Tuple[float, float] = (0, 0), **kwargs) -> Legend:
        return self.ax.legend(*args, loc='center left', bbox_to_anchor=(1+offset[0], .5+offset[1]), **kwargs)

    def remove_legend(self) -> None:
        self.ax.get_legend().remove() # type: ignore

    ######################## Quiver ########################
    
    def horizontal_quiver(self, limit: float) -> Quiver:
        return self.ax.quiver(-limit, 0, limit, 0, angles='xy', scale_units='xy', scale=.5)

    def vertical_quiver(self, limit: float) -> Quiver:
        return self.ax.quiver(0, -limit, 0, limit, angles='xy', scale_units='xy', scale=.5)
    
    ######################## Horizontal / Vertical Lines ########################

    def hline(
        self,
        y: float,
        xmin: float,
        xmax: float,
        linewidth: float = 1,
        **kwargs,
    ) -> LineCollection:
        return self.ax.hlines(y, xmin, xmax, linewidth=linewidth, **kwargs)
    
    def vline(
        self,
        x: float,
        ymin: float,
        ymax: float,
        linewidth: float = 1,
        **kwargs,
    ) -> LineCollection:
        return self.ax.vlines(x, ymin, ymax, linewidth=linewidth, **kwargs)

    ######################## Func ########################

    def line(
        self, 
        x: Any, 
        y: Any, 
        linewidth: float = 1, 
        markeredgecolor: Any = 'w',
        markeredgewidth: float = .75, 
        **kwargs,
    ) -> List[Line2D]:
        '''
        alpha: Optional[float] = None
        color: Any = None
        label: Optional[str] = None
        linestyle: str = '-'
            {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
        marker: Optional[str] = None
            {'o', 'X', '^'}
        '''
        return self.ax.plot(
            x, 
            y, 
            linewidth=linewidth, 
            markeredgecolor=markeredgecolor, 
            markeredgewidth=markeredgewidth, 
            **kwargs,
        )

    def line_with_band(
        self,
        x: Any, 
        center: Any, 
        diff: Any, 
        alpha_line: float = 1,
        alpha_band: float = .2,
        label: Optional[str] = None,
        linestyle: Optional[str] = None,
        linewidth: float = 1,
        marker: Optional[str] = None,
        markeredgecolor: Any = 'w',
        markeredgewidth: float = .75, 
        **kwargs,
    ) -> Tuple[List[Line2D], PolyCollection]:
        '''
        color: Any = None
        label: Optional[str] = None
        linestyle: str = '-'
        marker: Optional[str] = None,
        '''
        line = self.line(
            x, 
            center, 
            alpha=alpha_line, 
            linestyle=linestyle,
            linewidth=linewidth, 
            marker=marker,
            markeredgecolor=markeredgecolor, 
            markeredgewidth=markeredgewidth, 
            label=label,
            **kwargs,
        )
        polycollection = self.ax.fill_between(
            x, 
            center - diff, 
            center + diff, 
            alpha=alpha_band, 
            **kwargs,
        )
        return line, polycollection

    def hist(
        self, 
        data: Any, 
        element: Literal['bars', 'step', 'poly'] = 'step',
        stat: str = 'density',
        **kwargs,
    ) -> PltAxes:
        '''
        color: Any = None
        label: Optional[str] = None
        '''
        return sns.histplot(data=data, ax=self.ax, element=element, stat=stat, **kwargs)
    
    def scatter(self, x: Any, y: Any, alpha: float = .3, **kwargs) -> PathCollection:
        return self.ax.scatter(x, y, alpha=alpha, **kwargs)
    
    def scatter_o(
        self, 
        x: Any, 
        y: Any, 
        s: float = 40, 
        alpha: float = .3, 
        c: Any = None,
        edgecolors: Any = None,
        linewidths: float = .5, 
        **kwargs,
    ) -> PathCollection:
        '''
        c = [Figure.palette[0]]
        edgecolors = Figure.dark_palette[0]

        label: Optional[str] = None
        '''
        return self.scatter(
            x, 
            y, 
            s=s, 
            alpha=alpha, 
            c=c, 
            edgecolors=edgecolors, 
            linewidths=linewidths, 
            marker='o', 
            **kwargs,
        )
    
    def scatter_X(
        self, 
        x: Any, 
        y: Any, 
        s: float = 50, 
        alpha: float = .3, 
        c: Any = None,
        edgecolors: Any = None,
        linewidths: float = .5, 
        **kwargs,
    ) -> PathCollection:
        '''
        c = [Figure.palette[1]]
        edgecolors = Figure.dark_palette[1]

        label: Optional[str] = None
        '''
        return self.scatter(
            x, 
            y, 
            s=s, 
            alpha=alpha, 
            c=c, 
            edgecolors=edgecolors, 
            linewidths=linewidths, 
            marker='X', 
            **kwargs,
        )

    def contour(self, *args, **kwargs) -> QuadContourSet:
        '''
        args:
            Z: (M, N)
            or
            X: (M, N), Y: (M, N), Z: (M, N)
        '''
        return self.ax.contour(*args, **kwargs)
    
    def contour_boundary(self, *args, **kwargs) -> QuadContourSet:
        return self.contour(*args, levels=[-.0001, .0001], **kwargs)
    
    def contour_common_linewidth(self, *args, linewidth: float = 2, **kwargs) -> QuadContourSet:
        return self.contour(*args, linewidths=[linewidth], **kwargs)

    def contourf(self, *args, **kwargs) -> QuadContourSet:
        return self.ax.contourf(*args, **kwargs)
    
    def contourf_binary(self, *args, **kwargs) -> QuadContourSet:
        # colors = ['white', 'red']
        return self.contourf(*args, **kwargs, antialiased=False, levels=1)
    
        '''
        artists, labels: Tuple[List[Artist], List[str]] = cs.legend_elements()
        self.legend(artists, labels)
        '''

    def bar(self, x: Any, y: Any, width: float = .8, **kwargs) -> BarContainer:
        return self.ax.bar(x, y, width, **kwargs)
    
    def multibar(
        self, 
        x: Sequence, 
        ys: Sequence[Sequence], 
        total_width: float = .8, 
        **kwargs,
    ) -> List[BarContainer]:
        x_int = range(len(x))
        each_width = total_width / len(x)
        bars = []
        for i, y in enumerate(ys):
            x_int_plus_offset = [x + each_width * i for x in x_int]
            bar = self.bar(x_int_plus_offset, y, each_width, **kwargs)
            bars.append(bar)
        xticks_center = [t + each_width * len(x) / 2 for t in x_int]
        self.set_xticks(xticks_center)
        self.set_xticklabels(x)
        return bars
    
    def heatmap(self, map: Any, **kwargs) -> PltAxes:
        '''
        args:
            vmin: float
            vmax: float
            cmap
            linewidth: float
            cbar: bool = True
            square: bool = False
        '''
        assert len(map.shape) == 2
        return sns.heatmap(map, ax=self.ax, **kwargs)

    def imshow(
        self, 
        img: Any, 
        gray: bool = False, 
        vmin: float = 0,
        vmax: float = 1,
        remove_xticks: bool = True, 
        remove_yticks: bool = True
    ) -> None:
        if remove_xticks:
            self.remove_xticks()
        if remove_yticks:
            self.remove_yticks()
        l = len(img.shape)
        if l == 2:
            self.ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        elif l == 3:
            if gray:
                for i in img:
                    self.ax.imshow(i, cmap='gray', vmin=vmin, vmax=vmax)
            else:
                self.ax.imshow(img, vmin=vmin, vmax=vmax)
        elif l == 4:
            for i in img:
                self.ax.imshow(i, vmin=vmin, vmax=vmax)
        else:
            raise ValueError(l)


class Figure:
    palette: Tuple[Tuple[float, float, float], ...] = sns.color_palette('deep') # type: ignore
    dark_palette: Tuple[Tuple[float, float, float], ...] = sns.color_palette('dark') # type: ignore
    light_palette: Tuple[Tuple[float, float, float], ...] = sns.color_palette('pastel') # type: ignore
    color_map_blue_to_white_to_red: LinearSegmentedColormap \
        = LinearSegmentedColormap.from_list('blue_to_white_to_red (ascending)', [palette[0], 'white', palette[3]])
    
    @staticmethod
    def set_seaborn_theme() -> None:
        sns.set_theme()

    @staticmethod
    def set_font_scale(font_scale: float) -> None:
        '''
        From `seaborn/rcmod.py`,
            ```
            texts_base_context = {
                "font.size": 12,
                "axes.labelsize": 12,
                "axes.titlesize": 12,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 11,
                "legend.title_fontsize": 12,
            }
            ```
        '''
        sns.set_context('notebook', font_scale) # type: ignore
        plt.rcParams['figure.labelsize'] = 12 * font_scale
        plt.rcParams['figure.titlesize'] = 12 * font_scale

    @staticmethod
    def get_global_param(key: str) -> float:
        return plt.rcParams[key]

    @classmethod
    def set_tex(cls, luatex: bool = False) -> None:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{amsmath}
            \usepackage{amsfonts}
            \usepackage{bm}
        '''
        cls.set_font('tex')
        if luatex:
            # make Japanese label and TeX compatible
            cls.set_backend('lualatex')
            plt.rcParams['pgf.texsystem'] = 'lualatex'

    @classmethod
    def unset_tex(cls, luatex: bool = False) -> None:
        plt.rcParams['text.usetex'] = False
        if luatex:
            cls.set_backend('notebook')

    @staticmethod
    def set_backend(mode: Literal['notebook', 'lualatex']) -> None:
        if mode == 'notebook':
            backend = 'module://matplotlib_inline.backend_inline'
        elif mode == 'lualatex':
            backend = 'pgf'
        plt.rcParams['backend'] = backend

    @staticmethod
    def set_font(mode: Literal['default', 'tex', 'japanese']) -> None:
        if mode == 'default':
            plt.rcParams['font.family'] = ['sans-serif']
        elif mode == 'tex':
            plt.rcParams['font.family'] = 'cm'
        elif mode == 'japanese':
            # This is nearly equal to `plt.rcParams['font.family'] = 'IPAexGothic'`
            import japanize_matplotlib

    @staticmethod
    def set_mathfont(mode: Literal['default', 'tex']) -> None:
        # This is effective when `plt.rcParams['text.usetex'] = False`
        if mode == 'default':
            font = 'dejavusans'
        elif mode == 'tex':
            font = 'cm'
        plt.rcParams['mathtext.fontset'] = font

    @staticmethod
    def set_high_dpi() -> None:
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300

    @staticmethod
    def save(path: str, *paths: str) -> None:
        # For flexibly, do not use `self.fig.savefig(...)`
        p = os.path.join(path, *paths)
        plt.savefig(p, dpi=300, bbox_inches='tight', pad_inches=.025)

    @staticmethod
    def legend() -> None:
        plt.legend()

    @staticmethod
    def show() -> None:
        plt.show()
        plt.close()

    @staticmethod
    def close() -> None:
        plt.close()

    def __init__(
        self,
        n_rows: int = 1,
        n_cols: int = 1,
        figsize: Sequence[float] = (6.4, 4.8), 
        **kwargs,
    ) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.fig, self._axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, layout='constrained', **kwargs)
        self.axes = [[Axes(ax) for ax in row] for row in self._axes]
        
    def generate(self) -> Iterator[Axes]:
        # 0 1 2 3 4
        # 5 6 7 8 ...
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                yield self.axes[row][col]

    def set_suptitle(self, label: str, fontsize: Optional[float] = None) -> None:
        self.fig.suptitle(label, fontsize=fontsize or plt.rcParams['figure.labelsize'])

    def set_supxlabel(self, label: str, fontsize: Optional[float] = None) -> None:
        self.fig.supxlabel(label, fontsize=fontsize or plt.rcParams['figure.labelsize'])

    def set_supylabel(self, label: str, fontsize: Optional[float] = None) -> None:
        self.fig.supylabel(label, fontsize=fontsize or plt.rcParams['figure.labelsize'])

    @overload
    def set_axes_space(self, w_pad: float, h_pad: None) -> None:
        ...

    @overload
    def set_axes_space(self, w_pad: None, h_pad: float) -> None:
        ...

    @overload
    def set_axes_space(self, w_pad: float, h_pad: float) -> None:
        ...

    def set_axes_space(self, w_pad=None, h_pad=None) -> None:
        self.fig.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad)

    def set_legend_bottom(self, *args, offset: Tuple[float, float] = (0, 0), **kwargs) -> Legend:
        return self.fig.legend(*args, loc='upper center', bbox_to_anchor=(.5+offset[0], 0+offset[1]), **kwargs)
    
    def set_legend_right(self, *args, offset: Tuple[float, float] = (0, 0), **kwargs) -> Legend:
        return self.fig.legend(*args, loc='center left', bbox_to_anchor=(1+offset[0], .5+offset[1]), **kwargs)

    def set_colorbar(
        self,
        mappable: ScalarMappable, 
        label: Optional[str] = None, 
        label_fontsize: Optional[float] = None,
        labelpad: Optional[int] = None,
        tick_fontsize: Optional[float] = None, 
        pad: float = 0.01,
        **kwargs,
    ) -> Colorbar:
        cbar = self.fig.colorbar(mappable, ax=self._axes, pad=pad, **kwargs)
        if tick_fontsize:
            cbar.ax.tick_params(labelsize=tick_fontsize)
        if label:
            cbar.set_label(
                label, 
                labelpad=labelpad, 
                rotation=270, 
                size=label_fontsize or plt.rcParams['axes.labelsize'],
            )
        return cbar