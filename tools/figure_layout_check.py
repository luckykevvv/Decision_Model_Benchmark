"""Portable fixed-grid plot-area checks; full author QA remains separate."""
import json
from pathlib import Path
import numpy as np


def require_matplotlib_panel_alignment(fig,json_out=None,tolerance_pt=1.5,**kwargs):
    fig.canvas.draw()
    rects=np.array([ax.get_window_extent().bounds for ax in fig.axes if ax.get_visible()])*72/fig.dpi
    if len(rects)>1:
        assert np.ptp(rects[:,2])<=tolerance_pt,'Unequal fixed-grid plot widths'
        assert np.ptp(rects[:,3])<=tolerance_pt,'Unequal fixed-grid plot heights'
        for coordinate,extent in [(1,2),(0,3)]:
            for value in np.unique(np.round(rects[:,coordinate],4)):
                selected=rects[np.isclose(rects[:,coordinate],value)]
                if len(selected)>2:
                    order=np.argsort(selected[:,1-coordinate]);selected=selected[order]
                    gaps=selected[1:,1-coordinate]-(selected[:-1,1-coordinate]+selected[:-1,extent])
                    assert np.ptp(gaps)<=tolerance_pt,'Unequal fixed-grid gutters'
    report={'status':'portable fixed-grid geometry checks passed' if len(rects)>1 else 'single axes; comparison not applicable',
            'plot_area_rectangles_pt':rects.tolist(),'full_author_qa':'separate rendered glyph/collision inspection required'}
    if json_out:Path(json_out).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
