import os
import tempfile
from typing import List, Optional, Tuple

import gradio

import facefusion.processors.modules.face_swapper as face_swapper
from facefusion import face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, state_manager
from facefusion.common_helper import get_first
from facefusion.face_analyser import get_average_face, get_many_faces
from facefusion.face_selector import sort_faces_by_order
from facefusion.face_store import clear_static_faces
from facefusion.filesystem import filter_image_paths, is_image
from facefusion.types import Face, FaceSelectorOrder, VisionFrame
from facefusion.uis.types import File
from facefusion.vision import read_static_image, write_image

#
# A simplified, dedicated page for swapping MANY source faces onto MANY target
# faces in a single group photo. Each source image contributes one identity and
# those identities are mapped onto the faces found in the target, one-to-one,
# following the chosen ordering. The page deliberately uses the project defaults
# (hyperswap face swapper, yolo_face detector, box + occlusion masking) so it
# "just works" without any tuning.
#

MANY_TO_MANY_CSS =\
'''
.m2m-hero { text-align: center; padding: 1.25rem 1rem 0.5rem; }
.m2m-hero h1 { font-size: 1.6rem; font-weight: 700; margin: 0; }
.m2m-hero p { opacity: 0.7; margin: 0.35rem 0 0; font-size: 0.95rem; }
.m2m-step .label-wrap span { font-weight: 600; }
.m2m-swap-button button { font-size: 1.05rem !important; padding: 1rem !important; }
.m2m-status { min-height: 1.5rem; }
'''

MATCH_ORDER_CHOICES : List[Tuple[str, FaceSelectorOrder]] =\
[
	('Left to right', 'left-right'),
	('Right to left', 'right-left'),
	('Top to bottom', 'top-bottom'),
	('Largest to smallest', 'large-small')
]

SOURCE_FILE : Optional[gradio.File] = None
SOURCE_GALLERY : Optional[gradio.Gallery] = None
TARGET_IMAGE : Optional[gradio.Image] = None
MATCH_ORDER_DROPDOWN : Optional[gradio.Dropdown] = None
SWAP_BUTTON : Optional[gradio.Button] = None
CLEAR_BUTTON : Optional[gradio.Button] = None
RESULT_IMAGE : Optional[gradio.Image] = None
STATUS_MARKDOWN : Optional[gradio.Markdown] = None


def pre_check() -> bool:
	return True


def render() -> gradio.Blocks:
	global SOURCE_FILE
	global SOURCE_GALLERY
	global TARGET_IMAGE
	global MATCH_ORDER_DROPDOWN
	global SWAP_BUTTON
	global CLEAR_BUTTON
	global RESULT_IMAGE
	global STATUS_MARKDOWN

	with gradio.Blocks() as layout:
		gradio.HTML('<style>' + MANY_TO_MANY_CSS + '</style><div class="m2m-hero"><h1>Multi-Face Swap</h1><p>Drop a few face photos, drop a group photo, and swap everyone at once.</p></div>')
		with gradio.Row():
			with gradio.Column(scale = 5):
				SOURCE_FILE = gradio.File(
					label = '1.  Source faces  —  one clear photo per person',
					file_count = 'multiple',
					file_types = [ 'image' ],
					elem_classes = 'm2m-step'
				)
				SOURCE_GALLERY = gradio.Gallery(
					label = 'Source order',
					columns = 6,
					height = 140,
					object_fit = 'cover',
					allow_preview = False,
					show_label = True,
					visible = False
				)
			with gradio.Column(scale = 5):
				TARGET_IMAGE = gradio.Image(
					label = '2.  Group photo  —  the picture to swap faces into',
					type = 'filepath',
					elem_classes = 'm2m-step'
				)
		with gradio.Row():
			MATCH_ORDER_DROPDOWN = gradio.Dropdown(
				label = '3.  Match order  —  how source #1, #2, #3 line up with the faces in the photo',
				choices = MATCH_ORDER_CHOICES,
				value = 'left-right'
			)
		with gradio.Row():
			SWAP_BUTTON = gradio.Button(
				value = 'Swap faces',
				variant = 'primary',
				size = 'lg',
				elem_classes = 'm2m-swap-button'
			)
			CLEAR_BUTTON = gradio.Button(
				value = 'Clear',
				size = 'lg'
			)
		STATUS_MARKDOWN = gradio.Markdown(
			value = 'Upload your source faces and a group photo to get started.',
			elem_classes = 'm2m-status'
		)
		RESULT_IMAGE = gradio.Image(
			label = 'Result',
			interactive = False
		)
	return layout


def listen() -> None:
	SOURCE_FILE.change(update_source_gallery, inputs = SOURCE_FILE, outputs = SOURCE_GALLERY)
	SWAP_BUTTON.click(swap, inputs = [ SOURCE_FILE, TARGET_IMAGE, MATCH_ORDER_DROPDOWN ], outputs = [ RESULT_IMAGE, STATUS_MARKDOWN ])
	CLEAR_BUTTON.click(clear, outputs = [ SOURCE_FILE, TARGET_IMAGE, SOURCE_GALLERY, RESULT_IMAGE, STATUS_MARKDOWN ])


def run(ui : gradio.Blocks) -> None:
	ui.launch(favicon_path = 'facefusion.ico', inbrowser = state_manager.get_item('open_browser'))


def update_source_gallery(files : Optional[List[File]]) -> gradio.Gallery:
	source_paths = filter_image_paths([ file.name for file in files ]) if files else []

	if source_paths:
		gallery_value = [ (source_path, 'Source ' + str(index + 1)) for index, source_path in enumerate(source_paths) ]
		return gradio.Gallery(value = gallery_value, visible = True)
	return gradio.Gallery(value = None, visible = False)


def prepare_models() -> bool:
	common_modules =\
	[
		face_detector,
		face_landmarker,
		face_recognizer,
		face_classifier,
		face_masker
	]

	return all(module.pre_check() for module in common_modules) and face_swapper.pre_check()


def extract_source_faces(source_paths : List[str]) -> List[Face]:
	source_faces = []

	for source_path in source_paths:
		source_vision_frame = read_static_image(source_path)
		faces = sort_faces_by_order(get_many_faces([ source_vision_frame ]), 'large-small')

		if faces:
			source_face = get_average_face([ get_first(faces) ])
			if source_face:
				source_faces.append(source_face)
	return source_faces


def apply_best_settings() -> None:
	# box keeps the swap inside the face, occlusion lets hair / hands / other
	# faces in a busy group photo correctly cover the swapped region.
	face_mask_types = state_manager.get_item('face_mask_types') or []
	best_mask_types = list(face_mask_types)

	for mask_type in [ 'box', 'occlusion' ]:
		if mask_type not in best_mask_types:
			best_mask_types.append(mask_type)
	state_manager.set_item('face_mask_types', best_mask_types)


def swap(files : Optional[List[File]], target_path : Optional[str], match_order : FaceSelectorOrder):
	source_paths = filter_image_paths([ file.name for file in files ]) if files else []

	if not source_paths:
		yield None, '⚠️  Please upload at least one source face photo.'
		return
	if not is_image(target_path):
		yield None, '⚠️  Please upload a group photo to swap faces into.'
		return

	yield None, '⏳  Preparing models (first run downloads them, please wait)…'
	if not prepare_models():
		yield None, '❌  Could not prepare the required models. Check your connection and try again.'
		return

	apply_best_settings()
	clear_static_faces()

	yield None, '🔍  Detecting faces…'
	source_faces = extract_source_faces(source_paths)
	if not source_faces:
		yield None, '❌  No face was detected in your source photos. Use clearer, front-facing portraits.'
		return

	target_vision_frame = read_static_image(target_path)
	target_faces = sort_faces_by_order(get_many_faces([ target_vision_frame ]), match_order)
	if not target_faces:
		yield None, '❌  No face was detected in the group photo.'
		return

	swap_total = min(len(source_faces), len(target_faces))
	result_vision_frame : VisionFrame = target_vision_frame.copy()

	for index in range(swap_total):
		yield None, '✨  Swapping face ' + str(index + 1) + ' of ' + str(swap_total) + '…'
		result_vision_frame = face_swapper.swap_face(source_faces[index], target_faces[index], result_vision_frame)

	output_path = resolve_output_path()
	write_image(output_path, result_vision_frame)
	clear_static_faces()

	yield output_path, build_summary(len(source_faces), len(target_faces), swap_total)


def build_summary(source_total : int, target_total : int, swap_total : int) -> str:
	summary = '✅  Swapped ' + str(swap_total) + ' face' + ('s' if swap_total != 1 else '') + '.'

	if target_total > swap_total:
		summary += '  ' + str(target_total - swap_total) + ' extra face' + ('s' if target_total - swap_total != 1 else '') + ' in the photo had no source and were left unchanged.'
	if source_total > swap_total:
		summary += '  ' + str(source_total - swap_total) + ' extra source face' + ('s' if source_total - swap_total != 1 else '') + ' went unused (more sources than faces in the photo).'
	return summary


def resolve_output_path() -> str:
	output_directory = os.environ.get('GRADIO_TEMP_DIR') or tempfile.gettempdir()
	os.makedirs(output_directory, exist_ok = True)
	output_descriptor, output_path = tempfile.mkstemp(prefix = 'many_to_many_', suffix = '.png', dir = output_directory)
	os.close(output_descriptor)
	return output_path


def clear() -> Tuple[gradio.File, gradio.Image, gradio.Gallery, gradio.Image, gradio.Markdown]:
	clear_static_faces()
	return gradio.File(value = None), gradio.Image(value = None), gradio.Gallery(value = None, visible = False), gradio.Image(value = None), gradio.Markdown(value = 'Upload your source faces and a group photo to get started.')
