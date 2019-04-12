#!/usr/bin/env python3

import json
import os
import re
import time

# Relative path from this file to the notebook from which to generate.
IPYNB_PATH = os.path.join('..', 'notebooks', 'GANerator.ipynb')
# Relative path from this file to the source file to generate.
PY_PATH    = 'GANerator_generated.py'

# For this to work, all continuing plot lines after `plt.show` or
# `plt.imshow` should be marked with  a '# show' comment at the end of
# the line.
# This means any of those function spanning more than one line.
SHOW_PLOTS = False

EMPTY_RE = re.compile(r'^\s*$')
PLOT_RE  = re.compile(r'(^\s*plt\.(im)?show\()|(.*#\s*show$)')

FORMAT_RULES = '\n'.join((
    '- Document *every* parameter.',
    '- Use two spaces before the `#` sign if not at ' \
            'start of line (ignoring indentation).',
    '- Use one space after the `#` sign.',
    "- Space after `'param':`",
    '- If code spans more than one line, place the ' \
            'comment on comment-only lines _above_ the ' \
            'code.',
    '- After a section (marked by multiple `=` signs), ' \
            'leave an empty line if the next line is a ' \
            'comment referring code below (see above ' \
            'rule).',
    '- If code is on multiple lines, indent the ' \
            'continuation more.',
    # TODO fix below using indentation length
    '- Currently, no comments or empty lines in '
            'multi line code are supported.',
))


def filter_nb_only(line):
    return not (line.startswith('%matplotlib')
                or line.startswith('from IPython'))


def rstrip_spaces(line):
    i = len(line)
    while line[i - 1] == '\n':
        i -= 1
    return line[:i].rstrip() + line[i:]


def indent_content(line):
    if not SHOW_PLOTS and PLOT_RE.match(line):
        return '    # ' + rstrip_spaces(line)
    elif not EMPTY_RE.match(line):
        return '    ' + rstrip_spaces(line)
    else:
        return '\n'


def write_src(lines, src_file):
    """Write lines with four spaces indentation."""
    if type(lines) is not str:
        src_file.writelines(map(indent_content, lines))
    else:
        src_file.write(indent_content(lines))


def write_src_comments(comment_buffer, src_file):
    if comment_buffer:
        write_src(map(lambda line: '# ' + line + '\n',
                      comment_buffer), src_file)
        comment_buffer.clear()


def finish_src_code(code, src_file):
    if code:
        # for the last line in the code buffer,
        # remove everything starting from the last comma
        comma_ix = code[-1].rfind(',')
        # add closing bracket for `parser.add_argument`
        code[-1] = code[-1][:comma_ix] + ')\n'
        src_file.write(''.join(code))
        code.clear()


def process_parameters(source, src_file):
    i = 0
    while not source[i].startswith('params ='):
        i += 1
    write_src(source[:i], src_file)
    write_src('parser = argparse.ArgumentParser()\n', src_file)

    comment_re              = re.compile(r'^\s*# (.*)')
    code_with_comment_re    = re.compile(r"^\s*'(\S+?.*?)':\s+(.*),  # (.*)")
    code_without_comment_re = re.compile(r"^\s*'(\S+?.*?)':\s+(.*)")
    code_without_colon_re   = re.compile(r'^\s*(\S+.*)')

    comment_buffer = []
    code_buffer    = []
    for line in source[i + 1:-1]:  # all lines in the params dictionary

        m = comment_re.match(line)
        if m:
            finish_src_code(code_buffer, src_file)

            comment_buffer.append(m.groups()[0])
            continue

        if EMPTY_RE.match(line):
            finish_src_code(code_buffer, src_file)
            write_src_comments(comment_buffer, src_file)

            src_file.write('\n')
            continue

        m = code_with_comment_re.match(line)
        if m:
            finish_src_code(code_buffer, src_file)
            write_src_comments(comment_buffer, src_file)

            groups = m.groups()
            write_src(map(lambda line: line + '\n', (
                "parser.add_argument('--{}', nargs='+',".format(groups[0]),
                '        help="{}",'.format(groups[2]),
                "        default={})".format(groups[1])
            )), src_file)
            continue

        m = code_without_comment_re.match(line)
        if m and not code_buffer:
            # enter possible multi line
            groups = m.groups()
            write_src((
                "parser.add_argument('--{}', nargs='+',\n".format(groups[0]),
                '        help="{}",\n'.format(' '.join(comment_buffer)),
                '        default='
            ), src_file)
            comment_buffer.clear()
            code_buffer.append(groups[1])
            continue

        m = code_without_colon_re.match(line)
        if m and code_buffer:
            code_buffer.append(m.groups()[0])
            continue

        else:
            print('\n'.join((
                    'The parameters are not well formatted.',
                    '',
                    FORMAT_RULES
            )))
    finish_src_code(code_buffer, src_file)
    write_src(map(lambda line: line + '\n', (
        'params = vars(parser.parse_args())',
        'for key, val in params.items():',
        '    if type(val) is list:',
        '        if len(val) == 1:',
        '            params[key] = val[0]',
        '        else:',
        '            params[key] = tuple(val)',
    )), src_file)


def main():
    with open(os.path.join(os.path.dirname(__file__), IPYNB_PATH), 'r') as f:
        cells = json.load(f)['cells']

    params_processed = False
    with open(os.path.join(os.path.dirname(__file__), PY_PATH), 'w') as src_file:
        src_file.writelines(map(lambda line: line + '\n', (
            '#!/usr/bin/env python3',
            '',
            '# This source file was generated by ipynb_to_nb.py,',
            '# located at GANerator/src/ipynb_to_nb.py.',
            '# Time of creation: ' + time.strftime('%Y-%m-%d %H:%M:%S'),
            '',
        )))
        src_file.write('import argparse')
        src_file.writelines(filter(filter_nb_only, cells[0]['source']))
        src_file.write('\n\ndef main():')
        for cell in cells[1:-1]:
            src_file.write('\n')
            source = cell['source']
            if (not params_processed
                    and cell['metadata'].get('GANerator_parameters', False)):
                process_parameters(source, src_file)
                params_processed = True
                src_file.write('\n\n')
                continue
            write_src(source, src_file)
            src_file.write('\n\n')
        src_file.writelines(map(lambda line: line + '\n', (
            "\nif __name__ == '__main__':",
            '    main()\n'
        )))


if __name__ == '__main__':
    main()
