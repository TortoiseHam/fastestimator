# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any

from pylatex import LongTable, MultiColumn, NoEscape, Section
from pylatex.utils import bold

from fastestimator.trace.io.traceability import Traceability
from fastestimator.util.latex_util import Form, TextField, TextFieldBox
from fastestimator.util.traceability_util import traceable


@traceable()
class QmsTraceability(Traceability):
    """A GE internal version of the Traceability trace for QMS compliance.

    This introduces 3 additional sections: An Introduction, Abbreviations, and Revision History which can be populated
    after-the-fact.

    Args:
        save_path: Where to save the output files. Note that this will generate a new folder with the given name, into
            which the report and corresponding graphics assets will be written.
        extra_objects: Any extra objects which are not part of the Estimator, but which you want to capture in the
            summary report. One example could be an extra pipeline which performs pre-processing.
        n_definitions: How many rows to put in the definition section (or zero to disable).
        n_revisions: How many rows to put in the revision history section (or zero to disable).

    Raises:
        OSError: If graphviz is not installed.
        ValueError: If n_definitions is less than 0.
    """
    def __init__(self, save_path: str, extra_objects: Any = None, n_definitions: int = 6, n_revisions: int = 6):
        super().__init__(save_path, extra_objects)
        if n_definitions < 0:
            raise ValueError(f"n_definitions must be greater than zero, but got {n_definitions}.")
        self.n_definitions = n_definitions
        self.n_revisions = n_revisions

    def _write_body_content(self) -> None:
        """Write the main content of the file, plus GE specific entries.
        """
        with self.doc.create(Form()):
            self._document_intro()
            self._document_definitions()
            super()._write_body_content()
            self._document_revisions()

    def _document_intro(self) -> None:
        """Add a writable introduction field.
        """
        with self.doc.create(Section("Introduction")):
            self.doc.append(
                TextField(options=[
                    NoEscape(r'width=\textwidth'),
                    NoEscape(r'height=0.95\textheight'),
                    NoEscape('backgroundcolor={0.97 0.97 0.97}'),
                    'bordercolor=white',
                    'multiline=true',
                    'name=intro'
                ]))

    def _document_definitions(self) -> None:
        """Add a writable definitions.
        """
        if self.n_definitions > 0:
            with self.doc.create(Section("Definitions, Acronyms, and Abbreviations")):
                with self.doc.create(LongTable(r'|p{.23\textwidth}|p{0.7\textwidth}|', pos=['h!'],
                                               booktabs=True)) as tabular:
                    tabular.add_row((MultiColumn(size=1, align='|c|', data=bold("Term")),
                                     MultiColumn(size=1, align='c|', data=bold("Definition"))))
                    tabular.add_hline()
                    tabular.end_table_header()
                    tabular.add_row((MultiColumn(2, align='r', data='Continued on Next Page'), ))
                    tabular.add_hline()
                    tabular.end_table_footer()
                    tabular.end_table_last_footer()
                    for idx in range(self.n_definitions):
                        tabular.add_hline()
                        tabular.add_row((TextFieldBox(f"t{idx}"), TextFieldBox(f"d{idx}")))
            self.doc.append(NoEscape(r'\newpage'))

    def _document_revisions(self) -> None:
        """Add a writable revision history.
        """
        with self.doc.create(Section("Revision History")):
            with self.doc.create(
                    LongTable(r'|p{0.15\textwidth}|p{0.15\textwidth}|p{0.45\textwidth}|p{0.15\textwidth}|',
                              pos=['h!'],
                              booktabs=True)) as tabular:
                tabular.add_row((MultiColumn(size=1, align='|c|', data=bold("Version")),
                                 MultiColumn(size=1, align='c|', data=bold("Date")),
                                 MultiColumn(size=1, align='c|', data=bold("Reason For Change")),
                                 MultiColumn(size=1, align='c|', data=bold("Author"))))
                tabular.add_hline()
                tabular.end_table_header()
                tabular.add_row((MultiColumn(4, align='r', data='Continued on Next Page'), ))
                tabular.add_hline()
                tabular.end_table_footer()
                tabular.end_table_last_footer()
                for idx in range(self.n_definitions):
                    tabular.add_hline()
                    tabular.add_row((TextFieldBox(f"rv{idx}"),
                                     TextFieldBox(f"rd{idx}"),
                                     TextFieldBox(f"rr{idx}"),
                                     TextFieldBox(f"ra{idx}")))
