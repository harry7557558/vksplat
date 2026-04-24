import os
import json
import numpy as np
import scipy.stats
from datetime import datetime
from collections import OrderedDict


BREAKDOWN_PASSES = [
    ('ProjectionForward', 'project fwd'),
    ('CalculateIndexBufferOffset', 'idx offset'),
    ('GenerateKeys', 'gen keys'),
    ('Sort', 'sorting '),
    ('SortRTS', 'sorting '),
    ('ComputeTileRanges', 'tile ranges'),
    ('RasterizeForward', 'raster fwd'),
    ('CopyTrainImageToDevice', 'copy im 2 gpu'),
    ('ComputeSSIMGradient', 'ssim grad'),
    ('RasterizeBackward', 'raster bwd'),
    ('OptimizerStep', 'optim step'),
    ('FusedProjectionBackwardOptimizerStep', 'proj bwd opt'),
    ('DefaultPostBackward', 'densify '),
    ('MCMCPostBackward', 'densify ')
]


_html_components = []

builtin_print = print
def print(*args, **kwargs):
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    tag = kwargs.get('tag', 'span')
    builtin_print(*args, sep=sep, end=end)
    text = sep.join(map(str, args)) + end
    if text.endswith('\n'):
        text = text[:-1]
    html = f"<{tag}>{text.replace('\n', '<br/>')}</{tag}>"
    _html_components.append(html)

def tabulate_wrapper(table):
    from tabulate import tabulate
    builtin_print(tabulate(table, tablefmt="simple_grid"))
    
    import re
    html = tabulate(table, tablefmt='html')
    html = re.sub(r'[ ]*\n[ ]*', '<br/>', html)
    for i in range(2):
        html = re.sub(r'(</?\w+>)<br/>(</?\w+>)', r'\1\2', html)
        #html = html.replace("</tr><br/><tr>", "</tr><tr>")

    #from ansi2html import Ansi2HTMLConverter
    #html = Ansi2HTMLConverter(dark_bg=False, escaped=False, inline=True).convert(html)
    #html = html[html.find('<table>'):html.rfind('</pre>')]

    #html = re.sub(r'\033\[[\d\;]*m', '', html)
    #assert '\033' not in html
    
    html = re.sub(r'\033\[91m(.*?)\033\[m', r'<span style="color:red">\1</span>', html)
    html = re.sub(r'\033\[92m(.*?)\033\[m', r'<span style="color:green">\1</span>', html)
    html = re.sub(r'\033\[93m(.*?)\033\[m', r'<span style="color:darkorange">\1</span>', html)
    #html = re.sub(r'\033\[41m(.*?)\033\[m', r'<span style="color:white;background-color:red">\1</span>', html)
    #html = re.sub(r'\033\[42m(.*?)\033\[m', r'<span style="color:white;background-color:green">\1</span>', html)
    #html = re.sub(r'\033\[41m(.*?)\033\[m', r'<span style="color:red;font-weight:bold;font-style:italic">\1</span>', html)
    #html = re.sub(r'\033\[42m(.*?)\033\[m', r'<span style="color:green;font-weight:bold;font-style:italic">\1</span>', html)
    html = re.sub(r'\033\[41m(.*?)\033\[m', r'<span style="color:darkred;background-color:pink;font-weight:bold">\1</span>', html)
    html = re.sub(r'\033\[42m(.*?)\033\[m', r'<span style="color:darkgreen;background-color:aquamarine;font-weight:bold">\1</span>', html)
    html = re.sub(r'\033\[30;103m(.*?)\033\[m', r'<span style="color:brown;background-color:gold;font-weight:bold">\1</span>', html)
    html = re.sub(r'\033\[48;5;236m(.*?)\033\[m', r'<span style="color:white;background-color:gray;width:100%;display:inline-block;padding:0 0 0 0.25em">\1&nbsp;</span>', html)
    html = re.sub(r'\033\[38;2;(\d+);(\d+);(\d+)m(.*?)\033\[m', r'<span style="color:rgb(\1,\2,\3)">\4</span>', html)
    #print(re.findall(r'\033\[[\d\;]*m', html))
    
    _html_components.append(html)

def get_results(workdir):
    psnr, ssim, lpips, time, vram, gs = [np.nan] * 6
    breakdown = None

    try:
        with open(os.path.join(workdir, "train.json")) as fp:
            content = json.load(fp)
        time = content['time_elapsed']
        #vram = (content['peak_vram'] if 'peak_vram' in content else content['vram']) / 2**30
        vram = content['vram'] / 2**30  # to be fair with baseline, as torch.cuda.max_memory_allocated doesn't account for caching allocator
        breakdown = content['breakdown']
        gs = content['num_splats']
    except:
        pass

    try:
        with open(os.path.join(workdir, "eval.json")) as fp:
            content = json.load(fp)
        psnr = content['mean']['psnr']
        ssim = content['mean']['ssim']
        lpips = content['mean']['lpips_alex']
        # lpips = content['mean']['lpips_vgg']
    except:
        pass

    try:
        assert np.isnan(gs)
        with open(os.path.join(workdir, "splat.ply"), "rb") as fp:
            print(fp)
            header = fp.read(256).decode('ascii').strip().split('\n')
            assert header[0] == "ply"
            assert header[1] == "format binary_little_endian 1.0"
            line = header[2]  # element vertex 1000000
        gs = int(line.strip().split()[-1])
    except:
        pass

    if breakdown is not None:
        #print()
        breakdown_cleaned = OrderedDict()
        remaining_time = time
        for key, code in BREAKDOWN_PASSES:
            if key not in breakdown:
                breakdown_cleaned[code] = (0, float('nan'))
                continue
            count, step_time = breakdown[key]
            #print(key, step_time)
            remaining_time -= step_time
            if count == 0:
                if code in breakdown_cleaned and np.isfinite(breakdown_cleaned[code][1]):
                    continue
                step_time = float('nan')
            breakdown_cleaned[code] = (count, step_time)
        breakdown_cleaned[None] = (0, remaining_time)
        #print(remaining_time)
        if not remaining_time > 0.0:
            print("WARNING: non-positive remaining time")
        breakdown = breakdown_cleaned

    return (psnr, ssim, lpips, time, vram, gs), breakdown


def print_results(results, print_gs: bool):
    print()
    psnr, ssim, lpips, time, vram, gs = results
    print(f"{psnr:.2f}")
    print(f"{ssim:.3f}")
    print(f"{lpips:.3f}")
    print(round(time) if time==time else -1)
    print(f"{vram:.2f}")
    if print_gs:
        print(round(gs/1000) if gs==gs else -1)
    print()


def format_bar(x, w):
    bar = w * x
    bar = "█"*int(bar) + " ▏▎▍▌▋▊▉█"[int(8*(bar-int(bar))+0.5)]
    bar += ' '*(w-len(bar))
    return bar

def print_breakdown(breakdown, overall):
    print("Timing Breakdown:")
    hline = '\t'.join(["──────────", "────", "────"])
    print(hline)

    total = 0.0
    messages = []
    for code, (count, time) in breakdown.items():
        if not np.isfinite(time):
            continue
        if code is not None and count != 30000:
            messages.append(f"WARNING: count for `{code}` is {count}, not 30000")
        if code is None:
            if True: break
            print(hline)
            code = "(remaining)"

        bar = format_bar(time/overall, 24) + '▏'

        print(code, f"{time:.1f}", f"{100*time/overall:.1f}%", bar, sep='\t')

        if count > 0:
            total += time

    print('?       ', f"{overall-total:.1f} s", f"{100*(overall-total)/overall:.1f}%", sep='\t')
    print(hline)
    print('total   ', f"{total:.0f} s", f"{100*total/overall:.1f}%", sep='\t')
    print('\n'.join(messages))
    print()


def print_metrics_with_breakdown(workdir):
    result_dict = {}
    # for subdir_0 in ['1000000', '2000000', '3000000', 'default']:
    # for subdir_0 in ['default', '1000000', '2000000', '3000000']:
    for subdir_0 in sorted(os.listdir(workdir)):
        subdir = os.path.join(workdir, subdir_0)
        if not os.path.isdir(subdir):
            continue
        print('\n'+subdir_0+'\n')
        all = []
        for run_0 in sorted(os.listdir(subdir), key=lambda x: x[::-1]):
            scene = run_0.split('_')[-1]
            run = os.path.join(subdir, run_0)
            metrics, breakdown = get_results(run)
            print('\n>>>>', subdir_0, scene)
            print_results(metrics, 'default' in subdir_0)
            if breakdown is not None:
                print_breakdown(breakdown, metrics[3])
            if not np.isfinite(metrics).all():
                dt = datetime.now() - datetime.strptime(run_0[:run_0.rfind('_')], "%Y%m%d_%H%M%S")
                print(f"{dt.seconds}s elapsed\n")
            if not np.isnan(metrics).all():
                all.append(metrics)
            key = (subdir_0, scene)
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(metrics)
        mean = np.mean(all, 0)
        if np.isfinite(mean).any():
            print('\n>>>>', subdir_0, 'mean of', len(all), tag='h3')
            print_results(mean, 'default' in subdir_0)

    scene_names = ['bicycle', 'garden', 'stump', 'bonsai', 'counter', 'kitchen', 'room', '~']
    table = [['', ''] + scene_names]
    prev_key = None
    by_scene = None
    def append_cell(values):
        cell = []
        for row, label in zip(values.T, metrics):
            row = row.copy()
            if label == 'gs':
                row /= 1000
            confidence = 0.90
            df = len(row) - 1
            mean = np.mean(row)
            std = np.std(row)
            se = scipy.stats.sem(row)
            if std == 0.0:
                low, high = mean, mean
            else:
                low, high = scipy.stats.t.interval(confidence, df, loc=mean, scale=se)
            decimals = {
                'psnr': 2,
                'ssim': 3,
                'lpips': 3,
                'time': 0,
                'vram': 2,
                'gs': 0
            }[label]
            if not np.isfinite(high-low):
                continue
            low = f"{{:.{decimals}f}}".format(low)
            high = f"{{:.{decimals}f}}".format(high)
            shared = ""
            while len(low) > 0 and len(high) > 0 and low[0] == high[0]:
                shared = shared + low[0]
                low = low[1:]
                high = high[1:]
            if len(low) > 0 or len(high) > 0:
                shared = f"{shared}[{low}-{high}]"
            cell.append(shared)
        cell = '\n'.join(cell)
        table[-1].append(cell)
    def add_by_scene():
        if by_scene is None:
            return
        for metrics in zip(*by_scene.values()):
            metrics = np.stack(metrics)
            metrics = np.mean(metrics, 0)
        append_cell(metrics)
        pass
    for key, value in sorted(result_dict.items(), key=lambda x: (x[0][0], scene_names.index(x[0][1]))):
        metrics = 'psnr ssim lpips time vram gs'.split()
        #if key[0] != "default":
        #    metrics = metrics[:-1]
        if key[0] != prev_key:
            add_by_scene()
            by_scene = {}
            table.append([key[0], '\n'.join(metrics)])
            prev_key = key[0]
        value = np.stack(value)
        if key[1] not in by_scene:
            by_scene[key[1]] = []
        by_scene[key[1]].append(value)
        append_cell(value)
    add_by_scene()
    tabulate_wrapper(table)


def calc_p_values(data1, data2, fmts):
    from scipy.stats import wilcoxon
    p_values = []
    if isinstance(fmts, str) or fmts is None:
        fmts = [fmts] * len(data1.T)
    for m1, m2, fmt in zip(data1.T, data2.T, fmts):
        if np.isnan(m1+m2).any():
            p_values.append(np.nan)
            continue
        if fmt is not None:
            m1 = np.array([float(fmt.format(m)) for m in m1])
            m2 = np.array([float(fmt.format(m)) for m in m2])
        statistic, p_value = wilcoxon(m1, m2)
        p_values.append(p_value)
    return np.array(p_values)

def cmap_pvalue(p, is_worse):
    th = 0.017
    c = min(np.log(p)/np.log(th), 0.9999)
    return '38;2;'+(f'{int(96+160*c)};{int(96-32*c)};{int(96-32*c)}' if is_worse else
            f'{int(96-32*c)};{int(96+144*c)};{int(96+32*c)}')

def get_benchmark_results(workdir, prefix=''):
    all = {}
    for subdir_0 in sorted(os.listdir(workdir)):
        if not subdir_0.startswith(prefix):
            continue
        if prefix == '' and '-' in subdir_0:
            continue
        subdir = os.path.join(workdir, subdir_0)
        if not os.path.isdir(subdir):
            continue
        for run_0 in os.listdir(subdir) + [None]:
            run = os.path.join(subdir, run_0) if run_0 is not None else subdir
            metrics, breakdown = get_results(run)
            metrics = np.array(metrics)
            if np.isfinite(metrics).all():
                metrics[-1] *= 1e-3  # convert number of gaussians to k
                key = (subdir_0.lstrip(prefix), run_0.split('_')[-1]) if run_0 is not None else \
                    (None, subdir_0.split('_')[-1])
                all[key] = metrics
    batches = sorted(set([batch for (batch, dataset) in all.keys()]))
    for batch in batches:
        values = [val for (b, d), val in all.items() if b == batch]
        mean = np.mean(values, axis=0)
        all[(batch, None)] = mean
    return all

def compare_results(workdir1, prefix1, workdir2, prefix2):
    results1 = get_benchmark_results(workdir1, prefix1)
    results2 = get_benchmark_results(workdir2, prefix2)
    #batches = sorted(set([b for (b, d) in results1.keys()] + [b for (b, d) in results2.keys()]))
    #datasets = sorted(set([d for (b, d) in results1.keys() if d] + [d for (b, d) in results2.keys() if d]))
    batches = ['default', '1000000', '2000000', '3000000', None]
    datasets = ['bicycle', 'garden', 'stump', 'bonsai', 'counter', 'kitchen', 'room']
    # print('Batches:', batches)
    # print('Datasets:', datasets)
    # print()

    # higher better vs lower better
    signs = np.array([1.0, 1.0, -1.0, -1.0, -1.0, 0.0])
    fmts = ["{:.2f}", "{:.3f}", "{:.3f}", "{:.0f}", "{:.2f}", "{:.0f}"]
    thresholds = [0.2, 0.005, 0.005, 10, 0.05, 100]

    def format_table_str(metric, ref, _fmt=None, pvalue=False, is_worse=None):
        rows = []
        if is_worse is None:
            is_worse = [None] * len(metric)
        for m, r, fmt, s, th, iw in zip(metric, ref, fmts, signs, thresholds, is_worse):
            if _fmt:
                fmt = _fmt
            m = fmt.format(m)
            if m.startswith('-') and float(m) == 0.0:
                m = m[1:]
            r = float(fmt.format(r))
            diff = (float(m) - r)
            if np.isfinite(diff) or (pvalue and np.isfinite(float(m))):
                if pvalue: ecode = cmap_pvalue(float(m), iw)
                else: ecode = [[91, 41], [0, '30;103'], [92, 42]][int(np.sign(s*diff)+1)][int(abs(diff) > th)]
                if ecode: m = f"\033[{ecode}m{m}\033[m"
            rows.append(m)
        return '\n'.join(rows)

    header = [' ', ' '] + datasets + ['~']
    table1, table2 = [header+['diff', 'p-value']], [header]
    for batch in batches:
        key = {'1000000': 'MCMC\n1M', '2000000': 'MCMC\n2M', '3000000': 'MCMC\n3M', 'default': '~', None: '~'}[batch]
        label = '\n'.join(['PSNR', 'SSIM', 'LPIPS', 'Time[s]', 'Mem[GB]'] + ['#GS[k]']*(batch in ['default', None]))
        row1, row2 = [key, label], [key, label]
        data1, data2 = [], []
        valid1, valid2 = 1.0, 1.0
        for dataset in datasets+[None]:
            metric1 = results1.get((batch, dataset), [np.nan]*len(signs))
            metric2 = results2.get((batch, dataset), [np.nan]*len(signs))
            metric1, metric2 = np.array(metric1), np.array(metric2)
            valid1 *= np.linalg.norm(metric1) ** 1e-100
            valid2 *= np.linalg.norm(metric2) ** 1e-100
            if dataset is None:
                metric1, metric2 = valid1*metric1, valid2*metric2
            if batch not in ['default', None]:  # remove number of Gaussians
                metric1 = metric1[:-1]
                metric2 = metric2[:-1]
            data1.append(metric1)
            data2.append(metric2)
            row1.append(format_table_str(metric1, metric2))
            row2.append(format_table_str(metric2, metric1))

        row1.append(format_table_str(data1[-1]-data2[-1], [0.0]*100))
        #row2.append(format_table_str(data2[-1]-data1[-1], [0.0]*100))
        
        s = signs[:len(data1[-1])]
        is_worse = s*data1[-1] < s*data2[-1]
        data1, data2 = np.stack(data1)[:-1], np.stack(data2)[:-1]
        row1.append(format_table_str(
            calc_p_values(data1, data2, fmts), [np.nan]*100, "{:.2g}",
            pvalue=True, is_worse=is_worse
        ))

        if np.isfinite(data1).any() and np.isfinite(data2).any():
            table1.append(row1)
            table2.append(row2)

    if len(table1) > 1 and len(table2) > 1:
        print(workdir1, prefix1)
        tabulate_wrapper(table1)
        print()
        print(workdir2, prefix2)
        tabulate_wrapper(table2)


def cmap_percent(p):
    if not np.isfinite(p):
        return '0'
    c0, c1 = 240, 255
    c = c0+min(int((c1-c0+1)*p),c1-c0)
    return '38;5;'+str(c)

def get_benchmark_breakdown(workdir, prefix=''):
    batch_keys = {}
    all = {}
    for subdir_0 in sorted(os.listdir(workdir)):
        batch_key = subdir_0.lstrip(prefix)
        if not subdir_0.startswith(prefix):
            continue
        if prefix == '' and '-' in subdir_0:
            continue
        subdir = os.path.join(workdir, subdir_0)
        if not os.path.isdir(subdir):
            continue
        for run_0 in os.listdir(subdir) + [None]:
            run = os.path.join(subdir, run_0) if run_0 is not None else subdir
            metrics, breakdown = get_results(run)
            if breakdown is not None:
                batch_key_1 = (batch_key if run_0 is not None else None)
                if len(BREAKDOWN_PASSES) == 1:
                    del breakdown[None]
                    batch_keys[batch_key_1] = [*breakdown.keys()] + ['total']
                else:
                    batch_keys[batch_key_1] = [*breakdown.keys()][:-1] + ['?', 'total']
                timings = np.array([time for (count, time) in breakdown.values()] + [metrics[3]])
                key = (
                    batch_key_1,
                    (run_0 if run_0 is not None else subdir_0).split('_')[-1]
                )
                all[key] = timings
    batches = sorted(set([batch for (batch, dataset) in all.keys()]))
    for batch in batches:
        values = [val for (b, d), val in all.items() if b == batch]
        mean = np.mean(values, axis=0)
        all[(batch, None)] = mean
    return batch_keys, all

def compare_breakdown(workdir1, prefix1, workdir2, prefix2):
    keys1, results1 = get_benchmark_breakdown(workdir1, prefix1)
    keys2, results2 = get_benchmark_breakdown(workdir2, prefix2)
    #batches = sorted(set([b for (b, d) in results1.keys()] + [b for (b, d) in results2.keys()]))
    #datasets = sorted(set([d for (b, d) in results1.keys() if d] + [d for (b, d) in results2.keys() if d]))
    batches = ['default', '1000000', '2000000', '3000000', None]
    datasets = ['bicycle', 'garden', 'stump', 'bonsai', 'counter', 'kitchen', 'room']
    # print('Batches:', batches)
    # print('Datasets:', datasets)
    # print()

    BAR_WIDTH = 6

    def format_table_str(metric, ref, _fmt=None, _th=None, s=-1.0, pvalue=False, is_worse=None, percent=False):
        rows = []
        if is_worse is None:
            is_worse = [None] * len(metric)
        for m, r, iw in zip(metric, ref, is_worse):
            fmt = "{:.1f}" if _fmt is None else _fmt
            m = fmt.format(m)
            if m.startswith('-') and float(m.rstrip('%')) == 0.0:
                m = m[1:]
            r = float(fmt.format(r).rstrip('%'))
            diff = (float(m.rstrip('%')) - r)
            if np.isfinite(diff) or ((pvalue or percent) and np.isfinite(float(m))):
                th = 0.1*r if _th is None else _th
                # if percent: bar = format_bar(float(m)/np.amax(metric), BAR_WIDTH)
                if percent: bar = format_bar(float(m)/100, BAR_WIDTH)
                if pvalue: ecode = cmap_pvalue(float(m), iw)
                elif percent: ecode = cmap_percent((float(m)/np.amax(metric))**0.5)
                else: ecode = [[91, 41], [0, '30;103'], [92, 42]][int(np.sign(s*diff)+1)][int(abs(diff) > th)]
                if percent: m = m[:3]
                if ecode: m = f"\033[{ecode}m{m}\033[m"
                # if percent: m += f" \033[48;5;236m{bar}\033[m"
                if percent: m = f"\033[48;5;236m{bar}\033[m"
            rows.append(m)
        return '\n'.join(rows)

    for batch in batches:
        if batch not in keys1 or batch not in keys2:
            continue
        assert keys1[batch] == keys2[batch], (keys1[batch], keys2[batch])

        header = [' ', ' ', '%'] + datasets + ['~']
        table1, table2 = [header+['diff', 'dif %', 'p-value']], [header]

        key = {'1000000': 'MCMC\n1M', '2000000': 'MCMC\n2M', '3000000': 'MCMC\n3M', 'default': '~', None: batch}[batch]
        rows_keys = [*map(str, keys1[batch])]
        label = '\n'.join(rows_keys)
        row1, row2 = [key, label], [key, label]
        data1, data2 = [], []
        valid1, valid2 = [True]*len(rows_keys), [True]*len(rows_keys)
        for dataset in datasets+[None]:
            metric1 = results1.get((batch, dataset), [np.nan]*len(rows_keys))
            metric2 = results2.get((batch, dataset), [np.nan]*len(rows_keys))
            metric1, metric2 = np.array(metric1), np.array(metric2)
            valid1 &= np.isfinite(metric1)
            valid2 &= np.isfinite(metric2)
            if dataset is None:
               metric1 = np.where(valid1, metric1, [np.nan]*len(rows_keys))
               metric2 = np.where(valid2, metric2, [np.nan]*len(rows_keys))
            data1.append(metric1)
            data2.append(metric2)
            row1.append(format_table_str(metric1, metric2, _th=0.02*metric2[-1]))
            row2.append(format_table_str(metric2, metric1, _th=0.02*metric2[-1]))

        percent1 = 100*data1[-1][:-1]/data1[-1][-1]
        percent2 = 100*data2[-1][:-1]/data2[-1][-1]
        percent_1 = format_table_str(percent1, percent2*np.nan, "{:<4.1f}", 5, percent=True)
        percent_2 = format_table_str(percent2, percent1*np.nan, "{:<4.1f}", 5, percent=True)
        row1.insert(2, percent_1)
        row2.insert(2, percent_2)

        row1.append(format_table_str(data1[-1]-data2[-1], [0.0]*100, _th=0.02*data2[-1][-1]))
        row1.append(format_table_str(100*(data1[-1]/data2[-1]-1), [0.0]*100, "{:.1f}", 10))
        
        is_worse = data1[-1] > data2[-1]
        data1, data2 = np.stack(data1)[:-1], np.stack(data2)[:-1]
        row1.append(format_table_str(
            calc_p_values(data1, data2, "{:.1f}"), [np.nan]*100, "{:.2g}",
            pvalue=True, is_worse=is_worse
        ))

        def compare_row(row, mask1, mask2):
            mask1 = [*np.isfinite(mask1)] + [True]
            mask2 = [*~np.isfinite(mask2)] + [False]
            return row[:1] + ['\n'.join([
                [c, f"\033[93m{c}\033[m"][int(m2 and '\033' not in c)]
                for c, m1, m2 in zip(col.split('\n'), mask1, mask2)
                if m1
            ]) for col in row[1:]]
        mask = [i for i in range(len(data1)) if not np.isnan(data1[i]+data2[i]).all()]
        mask1 = np.array(data1)[mask].sum(0)
        mask2 = np.array(data2)[mask].sum(0)
        row1 = compare_row(row1, mask1, mask2)
        row2 = compare_row(row2, mask2, mask1)

        table1.append(row1)
        table2.append(row2)

        print(">>>> Timing breakdown of", key.replace('\n', ' ').replace('~', 'Default'), tag='h3')
        #print()
        print(workdir1, prefix1)
        tabulate_wrapper(table1)
        print(workdir2, prefix2)
        tabulate_wrapper(table2)
        print()



BASE_PATH = "D:\\harry\\outputs\\bench_m360_"  # adjust if needed

def main_breakdown():
    workdir = "20260405_eg2026_2"  # adjust if needed
    print_metrics_with_breakdown(BASE_PATH+workdir)


def main_compare(workdir1=None, workdir2=None):
    compares = [compare_results, compare_breakdown]

    workdir1, workdir2 = "20260405_eg2026_amd_2", "20260405_eg2026_2"  # adjust if needed

    workdir1, workdir2 = BASE_PATH+workdir1, BASE_PATH+workdir2
    for compare in compares:
        compare(workdir1, '', workdir2, '')

if __name__ == "__main__":
    #main_breakdown(); exit(0)
    main_compare(); #exit(0)
    
    if len(_html_components) > 0:
    
        from datetime import datetime
        _html_components.append("<br/><hr/><p>Generated: " + datetime.now().astimezone().isoformat() + "</p>")
        
        html_out_path = "D:\\\\harry\\outputs\\index.html"  # adjust if needed
        with open(html_out_path, "w", encoding='utf-8') as fp:
            fp.write('<!DOCTYPE HTML><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>')
            if True: fp.write("""<style>
body {
  margin: 20px 10px;
  font-size: 1em;
  #line-height: 1.2em;
  font-family: monospace;
}
h1, h2, h3, h4 {
    margin: 0;
}
table {
  border-collapse: collapse;
}
th, td {
  border: 1px solid black;
  padding: 5px 10px;
  vertical-align: top;
}
</style>""")
            fp.write("</head><body>")
            html = '<br/>\n'.join(_html_components)
            #html = html.replace("<table>", '<table style="border-collapse:collapse;font-family: monospace">')
            #html = html.replace("<td>", '<td style="border:1px solid black;padding:10px;vertical-align:top;line-height: 1.2em;">')
            fp.write(html)
            fp.write("</body></html>")
