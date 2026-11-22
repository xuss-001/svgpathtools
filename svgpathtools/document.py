"""(Experimental) replacement for import/export functionality.

This module contains the `Document` class, a container for a DOM-style 
document (e.g. svg, html, xml, etc.) designed to replace and improve 
upon the IO functionality of svgpathtools (i.e. the svg2paths and 
disvg/wsvg functions). 

An Historic Note:
     The functionality in this module is meant to replace and improve 
     upon the IO functionality previously provided by the the 
     `svg2paths` and `disvg`/`wsvg` functions. 

Example:
    Typical usage looks something like the following.

        >> from svgpathtools import Document
        >> doc = Document('my_file.html')
        >> for path in doc.paths():
        >>     # Do something with the transformed Path object.
        >>     foo(path)
        >>     # Inspect the raw SVG element, e.g. change its attributes
        >>     foo(path.element)
        >>     transform = result.transform
        >>     # Use the transform that was applied to the path.
        >>     foo(path.transform)
        >> foo(doc.tree)  # do stuff using ElementTree's functionality
        >> doc.display()  # display doc in OS's default application
        >> doc.save('my_new_file.html')

A Big Problem:  
    Derivatives and other functions may be messed up by 
    transforms unless transforms are flattened (and not included in 
    css)
"""

# External dependencies
from __future__ import division, absolute_import, print_function
import os
import xml.etree.ElementTree as etree
from xml.etree.ElementTree import Element, SubElement, register_namespace
from xml.dom.minidom import parseString
import warnings
from io import StringIO
from tempfile import gettempdir
from time import time
import numpy as np

# Internal dependencies
from .parser import parse_path
from .parser import parse_transform
from .svg_to_paths import (path2pathd, ellipse2pathd, line2pathd,
                           polyline2pathd, polygon2pathd, rect2pathd)
from .misctools import open_in_browser
from .path import transform, Path, is_path_segment

# To maintain forward/backward compatibility
try:
    string = basestring
except NameError:
    string = str
try:
    from os import PathLike
except ImportError:
    PathLike = string

# Let xml.etree.ElementTree know about the SVG namespace
SVG_NAMESPACE = {'svg': 'http://www.w3.org/2000/svg'}
register_namespace('svg', 'http://www.w3.org/2000/svg')

# THESE MUST BE WRAPPED TO OUTPUT ElementTree.element objects
CONVERSIONS = {'path': path2pathd,
               'circle': ellipse2pathd,
               'ellipse': ellipse2pathd,
               'line': line2pathd,
               'polyline': polyline2pathd,
               'polygon': polygon2pathd,
               'rect': rect2pathd}

CONVERT_ONLY_PATHS = {'path': path2pathd}

SVG_GROUP_TAG = 'svg:g'
SVG_USE_TAG = 'svg:use'
XLINK_NAMESPACE = 'http://www.w3.org/1999/xlink'
XLINK_HREF = '{{{0}}}href'.format(XLINK_NAMESPACE)


def _element_name(element):
    """Return the local tag name for an ElementTree element."""
    if not isinstance(element.tag, str):
        return None
    return element.tag.rsplit('}', 1)[-1]


def _element_label(element):
    name = _element_name(element) or 'element'
    element_id = element.get('id')
    if element_id:
        return '<{0} id={1!r}>'.format(name, element_id)
    return '<{0}>'.format(name)


def _use_xy_transform(use_elem):
    """Return the translation matrix contributed by a <use> x/y pair."""
    values = []
    for name in ('x', 'y'):
        value = use_elem.get(name)
        if value is None or value.strip() == '':
            value = 0.0
        else:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    'Invalid <use> {0} value {1!r} on {2}'
                    .format(name, use_elem.get(name), _element_label(use_elem)))
            if not np.isfinite(value):
                raise ValueError(
                    'Non-finite <use> {0} value {1!r} on {2}'
                    .format(name, use_elem.get(name), _element_label(use_elem)))
        values.append(value)

    transform = np.identity(3)
    transform[0, 2] = values[0]
    transform[1, 2] = values[1]
    return transform


def _build_id_index(root):
    """Map ids to elements, retaining duplicates so references can reject them."""
    id_index = {}
    for element in root.iter():
        element_id = element.get('id')
        if element_id:
            id_index.setdefault(element_id, []).append(element)
    return id_index


def flattened_paths(group, group_filter=lambda x: True,
                    path_filter=lambda x: True, path_conversions=CONVERSIONS,
                    group_search_xpath=SVG_GROUP_TAG,
                    strict_transform_parsing=False,
                    referenced_group_filter=None,
                    referenced_path_filter=None):
    """Returns the paths inside a group (recursively), expressing the
    paths in the base coordinates.

    Note that if the group being passed in is nested inside some parent
    group(s), we cannot take the parent group(s) into account, because
    xml.etree.Element has no pointer to its parent. You should use
    Document.flattened_paths_from_group(group) to flatten a specific nested group into
    the root coordinates.

    Args:
        group is an Element
        path_conversions (dict):
            A dictionary to convert from an SVG element to a path data
            string. Any element tags that are not included in this
            dictionary will be ignored (including the `path` tag). To
            only convert explicit path elements, pass in
            `path_conversions=CONVERT_ONLY_PATHS`.
        strict_transform_parsing (bool): If true, a ValueError is
            raised when a transform attribute contains invalid syntax;
            by default invalid transform substrings are skipped with
            an SVGSyntaxWarning.
        referenced_group_filter (callable): Group filter to use for groups
            reached through a <use> reference. This is needed internally by
            `flattened_paths_from_group()` so that referenced groups are not
            rejected by that function's traversal filter.
        referenced_path_filter (callable): Path filter to use for paths
            reached through a <use> reference, with the same internal purpose.
    """
    if not isinstance(group, Element):
        raise TypeError('Must provide an xml.etree.Element object. '
                        'Instead you provided {0}'.format(type(group)))

    # Stop right away if the group_selector rejects this group
    if not group_filter(group):
        warnings.warn('The input group [{}] (id attribute: {}) was rejected by the group filter'
                      .format(group, group.get('id')))
        return []

    if referenced_group_filter is None:
        referenced_group_filter = group_filter
    if referenced_path_filter is None:
        referenced_path_filter = path_filter

    id_index = _build_id_index(group)
    paths = []

    def parse_tf(element):
        return parse_transform(element.get('transform'),
                               strict=strict_transform_parsing)

    def attach_metadata(path, source_element, path_tf, use_chain):
        path.element = source_element
        path.transform = path_tf
        path.original_element = source_element
        path.source_element = source_element
        path.use_elements = tuple(use_chain)
        path.use_chain = path.use_elements
        use_element = path.use_elements[-1] if path.use_elements else None
        path.use_element = use_element
        path.instance_element = use_element
        path.use = use_element

    def emit_path(path_elem, parent_tf, use_chain):
        converter = path_conversions[_element_name(path_elem)]
        path_tf = parent_tf.dot(parse_tf(path_elem))
        path = transform(parse_path(converter(path_elem)), path_tf)
        attach_metadata(path, path_elem, path_tf, use_chain)
        paths.append(path)

    def resolve_use(use_elem):
        href = use_elem.get('href')
        href_attr = 'href'
        if href is None:
            href = use_elem.get(XLINK_HREF)
            href_attr = 'xlink:href'

        if href is None:
            raise ValueError(
                'Invalid <use> reference: {0} has neither href nor '
                'xlink:href'.format(_element_label(use_elem)))

        href = href.strip()
        if not href:
            raise ValueError(
                'Invalid <use> reference: {0} has an empty {1}'
                .format(_element_label(use_elem), href_attr))
        if not href.startswith('#'):
            raise ValueError(
                'External <use> references are not supported: {0} {1}={2!r}'
                .format(_element_label(use_elem), href_attr, href))

        target_id = href[1:]
        if not target_id:
            raise ValueError(
                'Invalid <use> reference: {0} has an empty fragment in {1}'
                .format(_element_label(use_elem), href_attr))

        matches = id_index.get(target_id, ())
        if not matches:
            raise ValueError(
                "Invalid <use> reference {0!r}: no element with that id "
                "exists in this SVG document (from {1})"
                .format(href, _element_label(use_elem)))
        if len(matches) > 1:
            raise ValueError(
                "Invalid <use> reference {0!r}: the id is ambiguous because "
                "it occurs {1} times (from {2})"
                .format(href, len(matches), _element_label(use_elem)))
        return href, matches[0]

    def expand_use(use_elem, parent_tf, active_targets, use_chain):
        href, target = resolve_use(use_elem)
        target_identity = id(target)
        if target_identity in active_targets:
            raise ValueError(
                'Circular <use> reference detected for {0!r} while expanding '
                '{1}'.format(href, _element_label(use_elem)))

        content_tf = parent_tf.dot(parse_tf(use_elem)).dot(
            _use_xy_transform(use_elem))
        expanded_use_chain = use_chain + (use_elem,)
        next_active_targets = active_targets | {target_identity}
        target_name = _element_name(target)

        if target_name == 'g':
            if not referenced_group_filter(target):
                return
            target_tf = content_tf.dot(parse_tf(target))
            flatten_container(target, target_tf, referenced_group_filter,
                              referenced_path_filter, expanded_use_chain,
                              next_active_targets)
        elif target_name == 'use':
            expand_use(target, content_tf, next_active_targets,
                       expanded_use_chain)
        elif target_name in CONVERSIONS or target_name in path_conversions:
            if (target_name in path_conversions
                    and referenced_path_filter(target)):
                emit_path(target, content_tf, expanded_use_chain)
        else:
            raise ValueError(
                'Unsupported <use> reference {0!r}: target {1} is not a basic '
                'shape or a <g>'.format(href, _element_label(target)))

    def flatten_container(container, container_tf, child_group_filter,
                          child_path_filter, use_chain, active_targets):
        group_children = {
            id(element) for element in
            container.iterfind(group_search_xpath, SVG_NAMESPACE)
        }

        for child in list(container):
            child_name = _element_name(child)
            if id(child) in group_children:
                if child_group_filter(child):
                    flatten_container(
                        child, container_tf.dot(parse_tf(child)),
                        child_group_filter, child_path_filter, use_chain,
                        active_targets)
            elif child_name == 'use':
                expand_use(child, container_tf, active_targets, use_chain)
            elif child_name in path_conversions and child_path_filter(child):
                emit_path(child, container_tf, use_chain)

    root_tf = parse_tf(group)
    flatten_container(group, root_tf, group_filter, path_filter, (),
                      frozenset())
    return paths


def flattened_paths_from_group(group_to_flatten, root, recursive=True,
                               group_filter=lambda x: True,
                               path_filter=lambda x: True,
                               path_conversions=CONVERSIONS,
                               group_search_xpath=SVG_GROUP_TAG,
                               strict_transform_parsing=False):
    """Flatten all the paths in a specific group.

    The paths will be flattened into the 'root' frame. Note that root
    needs to be an ancestor of the group that is being flattened.
    Otherwise, no paths will be returned.

    If `strict_transform_parsing` is true, a ValueError is raised when
    a transform attribute contains invalid syntax; by default invalid
    transform substrings are skipped with an SVGSyntaxWarning."""

    if not any(group_to_flatten is descendant for descendant in root.iter()):
        warnings.warn('The requested group_to_flatten is not a '
                      'descendant of root')
        # We will shortcut here, because it is impossible for any paths
        # to be returned anyhow.
        return []

    # We create a set of the unique IDs of each element that we wish to
    # flatten, if those elements are groups. Any groups outside of this
    # set will be skipped while we flatten the paths.
    desired_groups = set()
    if recursive:
        for group in group_to_flatten.iter():
            desired_groups.add(id(group))
    else:
        desired_groups.add(id(group_to_flatten))

    ignore_paths = set()
    # Use breadth-first search to find the path to the group that we care about
    if root is not group_to_flatten:
        search = [[root]]
        route = None
        while search:
            top = search.pop(0)
            frontier = top[-1]
            for child in frontier.iterfind(group_search_xpath, SVG_NAMESPACE):
                if child is group_to_flatten:
                    route = top
                    break
                future_top = list(top)
                future_top.append(child)
                search.append(future_top)

            if route is not None:
                for group in route:
                    # Add each group from the root to the parent of the desired group
                    # to the list of groups that we should traverse. This makes sure
                    # that paths will not stop before reaching the desired
                    # group.
                    desired_groups.add(id(group))
                    for key in path_conversions.keys():
                        for path_elem in group.iterfind('svg:'+key, SVG_NAMESPACE):
                            # Add each path in the parent groups to the list of paths
                            # that should be ignored. The user has not requested to
                            # flatten the paths of the parent groups, so we should not
                            # include any of these in the result.
                            ignore_paths.add(id(path_elem))
                break

        if route is None:
            raise ValueError('The group_to_flatten is not a descendant of the root!')

    def desired_group_filter(x):
        return (id(x) in desired_groups) and group_filter(x)

    def desired_path_filter(x):
        return (id(x) not in ignore_paths) and path_filter(x)

    return flattened_paths(root, desired_group_filter, desired_path_filter,
                           path_conversions, group_search_xpath,
                           strict_transform_parsing=strict_transform_parsing,
                           referenced_group_filter=group_filter,
                           referenced_path_filter=path_filter)


class Document:
    def __init__(self, filepath=None, strict_transform_parsing=False):
        """
        A container for a DOM-style SVG document.

        The `Document` class provides a simple interface to modify and analyze
        the path elements in a DOM-style document.  The DOM-style document is
        parsed into an ElementTree object (stored in the `tree` attribute).

        This class provides functions for extracting SVG data into Path objects.
        The output Path objects will be transformed based on their parent groups.

        Args:
            filepath (str or file-like): The filepath of the
                DOM-style object or a file-like object containing it.
            strict_transform_parsing (bool): If true, a ValueError is
                raised when a transform attribute containing invalid
                syntax is parsed (transforms are parsed lazily, by
                `paths()` and `paths_from_group()`); by default invalid
                transform substrings are skipped with an
                SVGSyntaxWarning.
        """
        self.strict_transform_parsing = strict_transform_parsing

        # strings are interpreted as file location everything else is treated as
        # file-like object and passed to the xml parser directly
        from_filepath = isinstance(filepath, string) or isinstance(filepath, PathLike)
        self.original_filepath = os.path.abspath(filepath) if from_filepath else None

        if filepath is None:
            self.tree = etree.ElementTree(Element('svg'))
        else:
            # parse svg to ElementTree object
            self.tree = etree.parse(filepath)

        self.root = self.tree.getroot()

    @classmethod
    def from_svg_string(cls, svg_string, strict_transform_parsing=False):
        """Constructor for creating a Document object from a string."""
        # wrap string into StringIO object
        svg_file_obj = StringIO(svg_string)
        # create document from file object
        return Document(svg_file_obj,
                        strict_transform_parsing=strict_transform_parsing)

    def paths(self, group_filter=lambda x: True,
              path_filter=lambda x: True, path_conversions=CONVERSIONS):
        """Returns a list of all paths in the document.

        Note that any transform attributes are applied before returning
        the paths.  Paths instantiated by <use> references are expanded into
        the document coordinate system. Their `element` attribute refers to
        the original shape, while `use_element` (also available as `use` and
        `instance_element`) refers to the <use> instance that created them.
        """
        return flattened_paths(
            self.tree.getroot(), group_filter, path_filter, path_conversions,
            strict_transform_parsing=self.strict_transform_parsing)

    def paths_from_group(self, group, recursive=True, group_filter=lambda x: True,
                         path_filter=lambda x: True, path_conversions=CONVERSIONS):
        if all(isinstance(s, string) for s in group):
            # If we're given a list of strings, assume it represents a
            # nested sequence
            group = self.get_group(group)
        elif not isinstance(group, Element):
            raise TypeError(
                'Must provide a list of strings that represent a nested '
                'group name, or provide an xml.etree.Element object. '
                'Instead you provided {0}'.format(group))

        if group is None:
            warnings.warn("Could not find the requested group!")
            return []

        return flattened_paths_from_group(
            group, self.tree.getroot(), recursive, group_filter, path_filter,
            path_conversions,
            strict_transform_parsing=self.strict_transform_parsing)

    def add_path(self, path, attribs=None, group=None):
        """Add a new path to the SVG."""

        # If not given a parent, assume that the path does not have a group
        if group is None:
            group = self.tree.getroot()

        # If given a list of strings (one or more), assume it represents
        # a sequence of nested group names
        elif len(group) > 0 and all(isinstance(elem, str) for elem in group):
            group = self.get_or_add_group(group)

        elif not isinstance(group, Element):
            raise TypeError(
                'Must provide a list of strings or an xml.etree.Element '
                'object. Instead you provided {0}'.format(group))

        else:
            # Make sure that the group belongs to this Document object
            if not self.contains_group(group):
                warnings.warn('The requested group does not belong to '
                              'this Document')

        # TODO: It might be better to use duck-typing here with a try-except
        if isinstance(path, Path):
            path_svg = path.d()
        elif is_path_segment(path):
            path_svg = Path(path).d()
        elif isinstance(path, string):
            # Assume this is a valid d-string.
            # TODO: Should we sanity check the input string?
            path_svg = path
        else:
            raise TypeError(
                'Must provide a Path, a path segment type, or a valid '
                'SVG path d-string. Instead you provided {0}'.format(path))

        if attribs is None:
            attribs = {}
        else:
            attribs = attribs.copy()

        attribs['d'] = path_svg

        return SubElement(group, 'path', attribs)

    def contains_group(self, group):
        return any(group is owned for owned in self.tree.iter())

    def get_group(self, nested_names, name_attr='id'):
        """Get a group from the tree, or None if the requested group
        does not exist. Use get_or_add_group(~) if you want a new group
        to be created if it did not already exist.

        `nested_names` is a list of strings which represent group names.
        Each group name will be nested inside of the previous group name.

        `name_attr` is the group attribute that is being used to
        represent the group's name. Default is 'id', but some SVGs may
        contain custom name labels, like 'inkscape:label'.

        Returns the request group. If the requested group did not
        exist, this function will return a None value.
        """
        group = self.tree.getroot()
        # Drill down through the names until we find the desired group
        while len(nested_names):
            prev_group = group
            next_name = nested_names.pop(0)
            for elem in group.iterfind(SVG_GROUP_TAG, SVG_NAMESPACE):
                if elem.get(name_attr) == next_name:
                    group = elem
                    break

            if prev_group is group:
                # The nested group could not be found, so we return None
                return None

        return group

    def get_or_add_group(self, nested_names, name_attr='id'):
        """Get a group from the tree, or add a new one with the given
        name structure.

        `nested_names` is a list of strings which represent group names.
        Each group name will be nested inside of the previous group name.

        `name_attr` is the group attribute that is being used to
        represent the group's name. Default is 'id', but some SVGs may
        contain custom name labels, like 'inkscape:label'.

        Returns the requested group. If the requested group did not
        exist, this function will create it, as well as all parent
        groups that it requires. All created groups will be left with
        blank attributes.

        """
        group = self.tree.getroot()
        # Drill down through the names until we find the desired group
        while len(nested_names):
            prev_group = group
            next_name = nested_names.pop(0)
            for elem in group.iterfind(SVG_GROUP_TAG, SVG_NAMESPACE):
                if elem.get(name_attr) == next_name:
                    group = elem
                    break

            if prev_group is group:
                # The group we're looking for does not exist, so let's
                # create the group structure
                nested_names.insert(0, next_name)

                while nested_names:
                    next_name = nested_names.pop(0)
                    group = self.add_group({'id': next_name}, group)
                # Now nested_names will be empty, so the topmost
                # while-loop will end
        return group

    def add_group(self, group_attribs=None, parent=None):
        """Add an empty group element to the SVG."""
        if parent is None:
            parent = self.tree.getroot()
        elif not self.contains_group(parent):
            warnings.warn('The requested group {0} does not belong to '
                          'this Document'.format(parent))

        if group_attribs is None:
            group_attribs = {}
        else:
            group_attribs = group_attribs.copy()

        return SubElement(parent, '{{{0}}}g'.format(
            SVG_NAMESPACE['svg']), group_attribs)

    def __repr__(self):
        return etree.tostring(self.tree.getroot()).decode()

    def pretty(self, **kwargs):
        return parseString(repr(self)).toprettyxml(**kwargs)

    def save(self, filepath, prettify=False, **kwargs):
        with open(filepath, 'w+') as output_svg:
            if prettify:
                output_svg.write(self.pretty(**kwargs))
            else:
                output_svg.write(repr(self))

    def display(self, filepath=None):
        """Displays/opens the doc using the OS's default application."""

        if filepath is None:
            if self.original_filepath is None: # created from empty Document
                orig_name, ext = 'unnamed', '.svg'
            else:
                orig_name, ext = \
                    os.path.splitext(os.path.basename(self.original_filepath))
            tmp_name = orig_name + '_' + str(time()).replace('.', '-') + ext
            filepath = os.path.join(gettempdir(), tmp_name)

        # write to a (by default temporary) file
        with open(filepath, 'w') as output_svg:
            output_svg.write(repr(self))

        open_in_browser(filepath)
