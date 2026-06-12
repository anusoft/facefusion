import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import gradio

import facefusion.processors.modules.face_enhancer.core as face_enhancer
import facefusion.processors.modules.face_swapper.core as face_swapper
from facefusion import face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, state_manager
from facefusion.common_helper import get_first
from facefusion.face_analyser import get_average_face, get_many_faces
from facefusion.face_selector import sort_faces_by_order
from facefusion.face_store import clear_static_faces
from facefusion.filesystem import filter_image_paths, is_image
from facefusion.types import Face, VisionFrame
from facefusion.uis.types import File
from facefusion.vision import read_static_image, write_image

#
# A simplified, dedicated page for swapping MANY faces onto MANY faces.
#
# Flow:
#   1. Drop source photo(s) -> every face found is shown as a removable chip.
#   2. Drop the target photo -> every face found is shown as a removable chip.
#   3. Delete any face you do not want on either side.
#   4. Click "Match all faces" -> source face #1 -> target face #1, #2 -> #2 ...
#      (both sides ordered left-to-right) then swap and clean up.
#
# Quality choices (research backed): inswapper_128 gives the best identity
# fidelity, and a GFPGAN face-enhancer pass removes the low-resolution "swapped"
# look, which is the single biggest quality win.
#

FACE_SWAPPER_MODEL = 'inswapper_128'
FACE_SWAPPER_PIXEL_BOOST = '256x256'
FACE_ENHANCER_MODEL = 'gfpgan_1.4'

MANY_TO_MANY_CSS =\
'''
.m2m-hero { text-align: center; padding: 1.25rem 1rem 0.5rem; }
.m2m-hero h1 { font-size: 1.6rem; font-weight: 700; margin: 0; }
.m2m-hero p { opacity: 0.7; margin: 0.35rem 0 0; font-size: 0.95rem; }
.m2m-swap-button button { font-size: 1.05rem !important; padding: 1rem !important; }
.m2m-status { min-height: 1.5rem; }
'''

SOURCE_FILE : Optional[gradio.File] = None
SOURCE_GALLERY : Optional[gradio.Gallery] = None
SOURCE_REMOVE_BUTTON : Optional[gradio.Button] = None
TARGET_FILE : Optional[gradio.Image] = None
TARGET_GALLERY : Optional[gradio.Gallery] = None
TARGET_REMOVE_BUTTON : Optional[gradio.Button] = None
ENHANCE_CHECKBOX : Optional[gradio.Checkbox] = None
MATCH_BUTTON : Optional[gradio.Button] = None
CLEAR_BUTTON : Optional[gradio.Button] = None
RESULT_IMAGE : Optional[gradio.Image] = None
STATUS_MARKDOWN : Optional[gradio.Markdown] = None

SOURCE_FACES_STATE : Optional[gradio.State] = None
TARGET_FACES_STATE : Optional[gradio.State] = None
TARGET_FRAME_STATE : Optional[gradio.State] = None
SOURCE_SELECTED_STATE : Optional[gradio.State] = None
TARGET_SELECTED_STATE : Optional[gradio.State] = None


def pre_check() -> bool:
	return True


def render() -> gradio.Blocks:
	global SOURCE_FILE, SOURCE_GALLERY, SOURCE_REMOVE_BUTTON
	global TARGET_FILE, TARGET_GALLERY, TARGET_REMOVE_BUTTON
	global ENHANCE_CHECKBOX, MATCH_BUTTON, CLEAR_BUTTON, RESULT_IMAGE, STATUS_MARKDOWN
	global SOURCE_FACES_STATE, TARGET_FACES_STATE, TARGET_FRAME_STATE, SOURCE_SELECTED_STATE, TARGET_SELECTED_STATE

	with gradio.Blocks() as layout:
		gradio.HTML('<style>' + MANY_TO_MANY_CSS + '</style><div class="m2m-hero"><h1>Multi-Face Swap</h1><p>Drop your faces and a group photo, remove the ones you don\'t want, then match everyone in one click.</p></div>')

		SOURCE_FACES_STATE = gradio.State([])
		TARGET_FACES_STATE = gradio.State([])
		TARGET_FRAME_STATE = gradio.State(None)
		SOURCE_SELECTED_STATE = gradio.State(None)
		TARGET_SELECTED_STATE = gradio.State(None)

		with gradio.Row():
			with gradio.Column():
				SOURCE_FILE = gradio.File(
					label = '1.  Source faces  —  the new identities',
					file_count = 'multiple',
					file_types = [ 'image' ]
				)
				SOURCE_GALLERY = gradio.Gallery(
					label = 'Detected source faces  —  click one, then Remove',
					columns = 5,
					height = 150,
					object_fit = 'cover',
					allow_preview = False,
					visible = False
				)
				SOURCE_REMOVE_BUTTON = gradio.Button(
					value = 'Remove selected source face',
					size = 'sm',
					visible = False
				)
			with gradio.Column():
				TARGET_FILE = gradio.Image(
					label = '2.  Target photo  —  the faces to replace',
					type = 'filepath'
				)
				TARGET_GALLERY = gradio.Gallery(
					label = 'Detected target faces  —  click one, then Remove',
					columns = 5,
					height = 150,
					object_fit = 'cover',
					allow_preview = False,
					visible = False
				)
				TARGET_REMOVE_BUTTON = gradio.Button(
					value = 'Remove selected target face',
					size = 'sm',
					visible = False
				)

		ENHANCE_CHECKBOX = gradio.Checkbox(
			label = 'Enhance faces after swapping (GFPGAN) — recommended',
			value = True
		)
		with gradio.Row():
			MATCH_BUTTON = gradio.Button(
				value = 'Match all faces',
				variant = 'primary',
				size = 'lg',
				elem_classes = 'm2m-swap-button'
			)
			CLEAR_BUTTON = gradio.Button(
				value = 'Clear',
				size = 'lg'
			)
		STATUS_MARKDOWN = gradio.Markdown(
			value = 'Drop source faces and a target photo to begin.',
			elem_classes = 'm2m-status'
		)
		RESULT_IMAGE = gradio.Image(
			label = 'Result',
			interactive = False
		)
	return layout


def listen() -> None:
	SOURCE_FILE.change(detect_source_faces, inputs = SOURCE_FILE, outputs = [ SOURCE_FACES_STATE, SOURCE_GALLERY, SOURCE_REMOVE_BUTTON, STATUS_MARKDOWN ])
	TARGET_FILE.change(detect_target_faces, inputs = TARGET_FILE, outputs = [ TARGET_FACES_STATE, TARGET_FRAME_STATE, TARGET_GALLERY, TARGET_REMOVE_BUTTON, STATUS_MARKDOWN ])

	SOURCE_GALLERY.select(store_selection, outputs = SOURCE_SELECTED_STATE)
	TARGET_GALLERY.select(store_selection, outputs = TARGET_SELECTED_STATE)

	SOURCE_REMOVE_BUTTON.click(remove_source_face, inputs = [ SOURCE_FACES_STATE, SOURCE_SELECTED_STATE ], outputs = [ SOURCE_FACES_STATE, SOURCE_GALLERY, SOURCE_SELECTED_STATE ])
	TARGET_REMOVE_BUTTON.click(remove_target_face, inputs = [ TARGET_FACES_STATE, TARGET_SELECTED_STATE ], outputs = [ TARGET_FACES_STATE, TARGET_GALLERY, TARGET_SELECTED_STATE ])

	MATCH_BUTTON.click(match_and_swap, inputs = [ SOURCE_FACES_STATE, TARGET_FACES_STATE, TARGET_FRAME_STATE, ENHANCE_CHECKBOX ], outputs = [ RESULT_IMAGE, STATUS_MARKDOWN ])
	CLEAR_BUTTON.click(clear, outputs = [ SOURCE_FILE, TARGET_FILE, SOURCE_GALLERY, SOURCE_REMOVE_BUTTON, TARGET_GALLERY, TARGET_REMOVE_BUTTON, RESULT_IMAGE, STATUS_MARKDOWN, SOURCE_FACES_STATE, TARGET_FACES_STATE, TARGET_FRAME_STATE, SOURCE_SELECTED_STATE, TARGET_SELECTED_STATE ])


def run(ui : gradio.Blocks) -> None:
	ui.launch(favicon_path = 'facefusion.ico', inbrowser = state_manager.get_item('open_browser'))


def crop_face(vision_frame : VisionFrame, face : Face) -> VisionFrame:
	frame_height, frame_width = vision_frame.shape[:2]
	start_x, start_y, end_x, end_y = face.bounding_box
	pad_x = (end_x - start_x) * 0.3
	pad_y = (end_y - start_y) * 0.3
	start_x = int(max(0, start_x - pad_x))
	start_y = int(max(0, start_y - pad_y))
	end_x = int(min(frame_width, end_x + pad_x))
	end_y = int(min(frame_height, end_y + pad_y))
	crop_vision_frame = vision_frame[start_y:end_y, start_x:end_x]
	return crop_vision_frame[:, :, ::-1]


def build_gallery(face_items : List[Dict[str, Any]], label_prefix : str) -> List[Tuple[VisionFrame, str]]:
	return [ (face_item.get('crop'), label_prefix + ' ' + str(index + 1)) for index, face_item in enumerate(face_items) ]


def detect_faces_in_frame(vision_frame : VisionFrame) -> List[Dict[str, Any]]:
	clear_static_faces()
	faces = sort_faces_by_order(get_many_faces([ vision_frame ]), 'left-right')
	return [ { 'face': face, 'crop': crop_face(vision_frame, face) } for face in faces ]


def detect_source_faces(files : Optional[List[File]]) -> Tuple[List[Dict[str, Any]], gradio.Gallery, gradio.Button, gradio.Markdown]:
	source_paths = filter_image_paths([ file.name for file in files ]) if files else []

	if not source_paths:
		return [], gradio.Gallery(value = None, visible = False), gradio.Button(visible = False), gradio.Markdown(value = 'Drop source faces and a target photo to begin.')

	face_items = []
	for source_path in source_paths:
		source_vision_frame = read_static_image(source_path)
		for face_item in detect_faces_in_frame(source_vision_frame):
			face_item['face'] = get_average_face([ face_item['face'] ])
			face_items.append(face_item)

	if not face_items:
		return [], gradio.Gallery(value = None, visible = False), gradio.Button(visible = False), gradio.Markdown(value = '❌  No face found in the source photos. Use clearer portraits.')
	return face_items, gradio.Gallery(value = build_gallery(face_items, 'Source'), visible = True), gradio.Button(visible = True), gradio.Markdown(value = '✅  Found ' + str(len(face_items)) + ' source face' + ('s' if len(face_items) != 1 else '') + '. Remove any you don\'t want.')


def detect_target_faces(target_path : Optional[str]) -> Tuple[List[Dict[str, Any]], Optional[VisionFrame], gradio.Gallery, gradio.Button, gradio.Markdown]:
	if not is_image(target_path):
		return [], None, gradio.Gallery(value = None, visible = False), gradio.Button(visible = False), gradio.Markdown(value = 'Drop source faces and a target photo to begin.')

	target_vision_frame = read_static_image(target_path)
	face_items = detect_faces_in_frame(target_vision_frame)

	if not face_items:
		return [], None, gradio.Gallery(value = None, visible = False), gradio.Button(visible = False), gradio.Markdown(value = '❌  No face found in the target photo.')
	return face_items, target_vision_frame, gradio.Gallery(value = build_gallery(face_items, 'Face'), visible = True), gradio.Button(visible = True), gradio.Markdown(value = '✅  Found ' + str(len(face_items)) + ' target face' + ('s' if len(face_items) != 1 else '') + '. Remove any you don\'t want.')


def store_selection(event : gradio.SelectData) -> int:
	return event.index


def remove_source_face(face_items : List[Dict[str, Any]], selected_index : Optional[int]) -> Tuple[List[Dict[str, Any]], gradio.Gallery, None]:
	face_items = remove_face_item(face_items, selected_index)
	return face_items, gradio.Gallery(value = build_gallery(face_items, 'Source'), visible = bool(face_items)), None


def remove_target_face(face_items : List[Dict[str, Any]], selected_index : Optional[int]) -> Tuple[List[Dict[str, Any]], gradio.Gallery, None]:
	face_items = remove_face_item(face_items, selected_index)
	return face_items, gradio.Gallery(value = build_gallery(face_items, 'Face'), visible = bool(face_items)), None


def remove_face_item(face_items : List[Dict[str, Any]], selected_index : Optional[int]) -> List[Dict[str, Any]]:
	if face_items and selected_index is not None and 0 <= selected_index < len(face_items):
		return [ face_item for index, face_item in enumerate(face_items) if index != selected_index ]
	return face_items


def apply_best_settings(do_enhance : bool) -> None:
	state_manager.set_item('face_swapper_model', FACE_SWAPPER_MODEL)
	state_manager.set_item('face_swapper_pixel_boost', FACE_SWAPPER_PIXEL_BOOST)

	if do_enhance:
		state_manager.set_item('face_enhancer_model', FACE_ENHANCER_MODEL)

	# box keeps the swap inside the face, occlusion lets hair / hands / other
	# faces in a busy group photo correctly cover the swapped region.
	face_mask_types = state_manager.get_item('face_mask_types') or []
	best_mask_types = list(face_mask_types)
	for mask_type in [ 'box', 'occlusion' ]:
		if mask_type not in best_mask_types:
			best_mask_types.append(mask_type)
	state_manager.set_item('face_mask_types', best_mask_types)


def prepare_models(do_enhance : bool) -> bool:
	common_modules =\
	[
		face_detector,
		face_landmarker,
		face_recognizer,
		face_classifier,
		face_masker
	]
	is_ready = all(module.pre_check() for module in common_modules) and face_swapper.pre_check()

	if do_enhance:
		is_ready = is_ready and face_enhancer.pre_check()
	return is_ready


def enhance_faces(vision_frame : VisionFrame) -> VisionFrame:
	clear_static_faces()
	faces = get_many_faces([ vision_frame ])

	for face in faces:
		vision_frame = face_enhancer.enhance_face(face, vision_frame)
	return vision_frame


def match_and_swap(source_items : List[Dict[str, Any]], target_items : List[Dict[str, Any]], target_vision_frame : Optional[VisionFrame], do_enhance : bool):
	if not source_items:
		yield None, '⚠️  Add at least one source face.'
		return
	if not target_items or target_vision_frame is None:
		yield None, '⚠️  Add a target photo with at least one face.'
		return

	yield None, '⏳  Preparing models (first run downloads them, please wait)…'
	apply_best_settings(do_enhance)
	if not prepare_models(do_enhance):
		yield None, '❌  Could not prepare the required models. Check your connection and try again.'
		return

	swap_total = min(len(source_items), len(target_items))
	result_vision_frame = target_vision_frame.copy()

	for index in range(swap_total):
		yield None, '✨  Swapping face ' + str(index + 1) + ' of ' + str(swap_total) + '…'
		result_vision_frame = face_swapper.swap_face(source_items[index]['face'], target_items[index]['face'], result_vision_frame)

	if do_enhance:
		yield None, '🎨  Enhancing faces…'
		result_vision_frame = enhance_faces(result_vision_frame)

	output_path = resolve_output_path()
	write_image(output_path, result_vision_frame)
	clear_static_faces()

	yield output_path, build_summary(len(source_items), len(target_items), swap_total)


def build_summary(source_total : int, target_total : int, swap_total : int) -> str:
	summary = '✅  Matched and swapped ' + str(swap_total) + ' face' + ('s' if swap_total != 1 else '') + ', left to right.'

	if target_total > swap_total:
		summary += '  ' + str(target_total - swap_total) + ' target face' + ('s' if target_total - swap_total != 1 else '') + ' had no source and stayed unchanged.'
	if source_total > swap_total:
		summary += '  ' + str(source_total - swap_total) + ' source face' + ('s' if source_total - swap_total != 1 else '') + ' went unused.'
	return summary


def resolve_output_path() -> str:
	output_directory = os.environ.get('GRADIO_TEMP_DIR') or tempfile.gettempdir()
	os.makedirs(output_directory, exist_ok = True)
	output_descriptor, output_path = tempfile.mkstemp(prefix = 'many_to_many_', suffix = '.png', dir = output_directory)
	os.close(output_descriptor)
	return output_path


def clear() -> Tuple[Any, ...]:
	clear_static_faces()
	return (
		gradio.File(value = None),
		gradio.Image(value = None),
		gradio.Gallery(value = None, visible = False),
		gradio.Button(visible = False),
		gradio.Gallery(value = None, visible = False),
		gradio.Button(visible = False),
		gradio.Image(value = None),
		gradio.Markdown(value = 'Drop source faces and a target photo to begin.'),
		[],
		[],
		None,
		None,
		None
	)
