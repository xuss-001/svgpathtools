from __future__ import division, absolute_import, print_function
import unittest
import xml.etree.ElementTree as etree

from svgpathtools import Document, CONVERT_ONLY_PATHS


SVG_OPEN = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'xmlns:xlink="http://www.w3.org/1999/xlink">'
)


class TestUseExpansion(unittest.TestCase):

    def assert_complex_almost_equal(self, actual, expected):
        self.assertAlmostEqual(actual.real, expected.real)
        self.assertAlmostEqual(actual.imag, expected.imag)

    def get_path_for_use(self, paths, use_id):
        return next(path for path in paths
                    if path.use_element is not None
                    and path.use_element.get('id') == use_id)

    def test_href_and_xlink_href_expand_basic_shapes(self):
        doc = Document.from_svg_string(
            SVG_OPEN +
            '<defs><path id="source" d="M0 0 L10 0"/></defs>'
            '<use id="href-use" href="#source" x="2" y="3"/>'
            '<use id="xlink-use" xlink:href="#source" x="20" y="30"/>'
            '</svg>'
        )

        paths = doc.paths()
        self.assertEqual(len(paths), 2)

        href_path = self.get_path_for_use(paths, 'href-use')
        xlink_path = self.get_path_for_use(paths, 'xlink-use')

        self.assert_complex_almost_equal(href_path.start, 2 + 3j)
        self.assert_complex_almost_equal(href_path.end, 12 + 3j)
        self.assert_complex_almost_equal(xlink_path.start, 20 + 30j)
        self.assert_complex_almost_equal(xlink_path.end, 30 + 30j)

    def test_translate_and_rotate_composition_order(self):
        doc = Document.from_svg_string(
            SVG_OPEN +
            '<defs><path id="source" d="M0 0 L10 0"/></defs>'
            '<use id="translate-rotate" href="#source" x="5" y="7" '
            'transform="translate(3 4) rotate(90)"/>'
            '<use id="rotate-translate" href="#source" x="5" y="7" '
            'transform="rotate(90) translate(3 4)"/>'
            '</svg>'
        )

        paths = doc.paths()
        translate_rotate = self.get_path_for_use(paths, 'translate-rotate')
        rotate_translate = self.get_path_for_use(paths, 'rotate-translate')

        self.assert_complex_almost_equal(translate_rotate.start, -4 + 9j)
        self.assert_complex_almost_equal(translate_rotate.end, -4 + 19j)
        self.assert_complex_almost_equal(rotate_translate.start, -11 + 8j)
        self.assert_complex_almost_equal(rotate_translate.end, -11 + 18j)

    def test_same_shape_referenced_by_multiple_uses_is_independent(self):
        doc = Document.from_svg_string(
            SVG_OPEN +
            '<defs><path id="shared" d="M0 0 L10 0"/></defs>'
            '<use id="first" href="#shared"/>'
            '<use id="second" href="#shared"/>'
            '</svg>'
        )

        paths = doc.paths()
        first = self.get_path_for_use(paths, 'first')
        second = self.get_path_for_use(paths, 'second')

        self.assertIsNot(first, second)
        self.assertIsNot(first[0], second[0])
        self.assertIs(first.element, second.element)
        self.assertIs(first.original_element, second.original_element)
        self.assertIs(first.element, first.original_element)

        first[0].start = 100 + 100j
        self.assert_complex_almost_equal(second[0].start, 0j)

    def test_nested_use_expands_through_group_transforms(self):
        doc = Document.from_svg_string(
            SVG_OPEN +
            '<defs>'
            '<path id="source" d="M0 0 L10 0"/>'
            '<use id="inner" href="#source" x="2" y="3"/>'
            '<g id="group" transform="translate(10 20)">'
            '<use id="middle" href="#inner" x="4" y="5" '
            'transform="rotate(90)"/>'
            '</g>'
            '</defs>'
            '<use id="outer" href="#group" x="6" y="7" '
            'transform="translate(100 200)"/>'
            '</svg>'
        )

        paths = doc.paths()
        self.assertEqual(len(paths), 1)
        path = paths[0]

        self.assert_complex_almost_equal(path.start, 108 + 233j)
        self.assert_complex_almost_equal(path.end, 108 + 243j)
        self.assertEqual(path.element.get('id'), 'source')
        self.assertEqual([use.get('id') for use in path.use_elements],
                         ['outer', 'middle', 'inner'])
        self.assertEqual(path.use_element.get('id'), 'inner')
        self.assertIs(path.use, path.use_element)
        self.assertIs(path.instance_element, path.use_element)
        self.assertIs(path.source_element, path.element)

    def test_use_group_and_source_transforms_are_all_applied(self):
        doc = Document.from_svg_string(
            SVG_OPEN +
            '<defs>'
            '<g id="source-group" transform="translate(10 20)">'
            '<path id="source" d="M0 0 L2 0" transform="rotate(90)"/>'
            '</g>'
            '</defs>'
            '<g transform="scale(2)">'
            '<use id="instance" href="#source-group" x="3" y="4" '
            'transform="translate(5 6)"/>'
            '</g>'
            '</svg>'
        )

        path = doc.paths()[0]
        self.assert_complex_almost_equal(path.start, 36 + 60j)
        self.assert_complex_almost_equal(path.end, 36 + 64j)

    def test_uninstantiated_defs_content_is_not_returned(self):
        doc = Document.from_svg_string(
            SVG_OPEN +
            '<defs><path id="unused" d="M0 0 L10 0"/></defs>'
            '<path id="visible" d="M20 20 L30 30"/>'
            '</svg>'
        )

        paths = doc.paths()
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].element.get('id'), 'visible')
        self.assertIsNone(paths[0].use_element)
        self.assertEqual(paths[0].use_elements, ())

    def test_existing_filters_still_apply_to_use_sources(self):
        doc = Document.from_svg_string(
            SVG_OPEN +
            '<defs>'
            '<path id="keep" d="M0 0 L1 0"/>'
            '<path id="drop" d="M0 0 L2 0"/>'
            '<circle id="circle" cx="1" cy="1" r="1"/>'
            '<g id="hidden-group"><path id="grouped" d="M0 0 L3 0"/></g>'
            '</defs>'
            '<use id="keep-use" href="#keep"/>'
            '<use id="drop-use" href="#drop"/>'
            '<use id="circle-use" href="#circle"/>'
            '<use id="group-use" href="#hidden-group"/>'
            '</svg>'
        )

        paths = doc.paths(path_filter=lambda elem: elem.get('id') == 'keep')
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].element.get('id'), 'keep')

        paths = doc.paths(path_conversions=CONVERT_ONLY_PATHS)
        self.assertEqual(
            {path.element.get('id') for path in paths},
            {'keep', 'drop', 'grouped'})

        paths = doc.paths(
            group_filter=lambda elem: elem.get('id') != 'hidden-group')
        self.assertNotIn('grouped', {path.element.get('id') for path in paths})

    def test_paths_from_group_expands_nested_use(self):
        doc = Document.from_svg_string(
            SVG_OPEN +
            '<path id="source" d="M0 0 L10 0"/>'
            '<g id="selection"><use id="instance" href="#source" x="4"/></g>'
            '</svg>'
        )

        paths = doc.paths_from_group(['selection'])
        self.assertEqual(len(paths), 1)
        self.assert_complex_almost_equal(paths[0].start, 4 + 0j)

    def test_document_tree_is_not_modified(self):
        doc = Document.from_svg_string(
            SVG_OPEN +
            '<defs><path id="source" d="M0 0 L10 0"/></defs>'
            '<use id="instance" href="#source" x="4" y="5" '
            'transform="rotate(30)"/>'
            '</svg>'
        )
        before = etree.tostring(doc.root)

        doc.paths()

        self.assertEqual(etree.tostring(doc.root), before)

    def test_invalid_missing_and_external_references_raise(self):
        invalid_docs = [
            SVG_OPEN + '<use id="no-href"/></svg>',
            SVG_OPEN + '<use id="missing" href="#does-not-exist"/></svg>',
            SVG_OPEN + '<use id="external" href="other.svg#shape"/></svg>',
            SVG_OPEN + '<use id="external-xlink" '
                       'xlink:href="https://example.com/other.svg#shape"/>'
                       '</svg>',
        ]

        for svg in invalid_docs:
            with self.assertRaises(ValueError):
                Document.from_svg_string(svg).paths()

    def test_circular_references_raise(self):
        svg = (
            SVG_OPEN +
            '<defs>'
            '<use id="a" href="#b"/>'
            '<use id="b" href="#a"/>'
            '</defs>'
            '<use id="outer" href="#a"/>'
            '</svg>'
        )

        with self.assertRaisesRegex(ValueError, '[Cc]ircular'):
            Document.from_svg_string(svg).paths()


if __name__ == '__main__':
    unittest.main()
