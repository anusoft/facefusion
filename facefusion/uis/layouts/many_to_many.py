import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio
import numpy

import facefusion.processors.modules.face_enhancer.core as face_enhancer
import facefusion.processors.modules.face_swapper.core as face_swapper
from facefusion import face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, state_manager
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
#   1. Drop source photo(s) -> each detected face becomes "Source 1, 2, 3 ...".
#   2. Drop the target photo -> each detected face is shown with a dropdown.
#   3. For every target face, pick which source replaces it (auto-matched by
#      order, change any you like, or keep the original).
#   4. Click "Swap faces".
#
# Quality choices (research backed): inswapper_128 gives the best identity
# fidelity, and a GFPGAN face-enhancer pass removes the low-resolution "swapped"
# look, which is the single biggest quality win.
#

# 'face'  -> inswapper_128 (arcface crop): sharpest identity, eyebrows-to-chin.
# 'wide'  -> hififace_unofficial_256 (mtcnn_512 crop): covers forehead + jaw up
#            to the hairline, the closest to a whole-head look these models can
#            do. Note: NONE of the swapper models can replace the source's hair.
SWAP_AREA_SET =\
{
	'face': { 'model': 'inswapper_128', 'pixel_boost': '256x256', 'mask_types': [ 'box', 'occlusion' ] },
	'wide': { 'model': 'hififace_unofficial_256', 'pixel_boost': '256x256', 'mask_types': [ 'box' ] }
}
FACE_ENHANCER_MODEL = 'gfpgan_1.4'
KEEP_ORIGINAL = -1

MANY_TO_MANY_CSS =\
'''
.m2m-hero { text-align: center; padding: 1.25rem 1rem 0.5rem; }
.m2m-hero h1 { font-size: 1.6rem; font-weight: 700; margin: 0; }
.m2m-hero p { opacity: 0.7; margin: 0.35rem 0 0; font-size: 0.95rem; }
.m2m-swap-button button { font-size: 1.05rem !important; padding: 1rem !important; }
.m2m-status { min-height: 1.5rem; }
.m2m-face-card img { border-radius: 0.375rem; }
'''

SOURCE_FILE : Optional[gradio.File] = None
TARGET_FILE : Optional[gradio.Image] = None
SWAP_AREA_RADIO : Optional[gradio.Radio] = None
ENHANCE_CHECKBOX : Optional[gradio.Checkbox] = None
SWAP_BUTTON : Optional[gradio.Button] = None
CLEAR_BUTTON : Optional[gradio.Button] = None
RESULT_IMAGE : Optional[gradio.Image] = None
STATUS_MARKDOWN : Optional[gradio.Markdown] = None

SOURCE_FACES_STATE : Optional[gradio.State] = None
TARGET_FACES_STATE : Optional[gradio.State] = None
TARGET_FRAME_STATE : Optional[gradio.State] = None


def pre_check() -> bool:
	return True


def render() -> gradio.Blocks:
	global SOURCE_FILE, TARGET_FILE, SWAP_AREA_RADIO, ENHANCE_CHECKBOX, SWAP_BUTTON, CLEAR_BUTTON, RESULT_IMAGE, STATUS_MARKDOWN
	global SOURCE_FACES_STATE, TARGET_FACES_STATE, TARGET_FRAME_STATE

	with gradio.Blocks() as layout:
		gradio.HTML('<style>' + MANY_TO_MANY_CSS + '</style><div class="m2m-hero"><h1>Multi-Face Swap</h1><p>Add your source faces, drop a target photo, then pick which source replaces each target face.</p></div>')

		SOURCE_FACES_STATE = gradio.State([])
		TARGET_FACES_STATE = gradio.State([])
		TARGET_FRAME_STATE = gradio.State(None)

		with gradio.Row():
			with gradio.Column():
				SOURCE_FILE = gradio.File(
					label = '1.  Source faces  —  the new identities',
					file_count = 'multiple',
					file_types = [ 'image' ]
				)
				render_source_palette()
			with gradio.Column():
				TARGET_FILE = gradio.Image(
					label = '2.  Target photo  —  the faces to replace',
					type = 'filepath'
				)

		gradio.Markdown('### 3.  Match each target face to a source')
		render_target_matcher()

		SWAP_AREA_RADIO = gradio.Radio(
			label = 'Swap area  (hair is always kept from the target — these models can\'t replace hair)',
			choices = [ ('Face — sharpest match (eyebrows to chin)', 'face'), ('Wide — forehead + jaw, up to the hairline', 'wide') ],
			value = 'face'
		)
		ENHANCE_CHECKBOX = gradio.Checkbox(
			label = 'Enhance faces after swapping (GFPGAN) — recommended',
			value = True
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
			value = 'Add source faces and a target photo to begin.',
			elem_classes = 'm2m-status'
		)
		RESULT_IMAGE = gradio.Image(
			label = 'Result',
			interactive = False
		)
	return layout


def render_source_palette() -> None:
	@gradio.render(inputs = SOURCE_FACES_STATE)
	def render_palette(source_items : List[Dict[str, Any]]) -> None:
		if not source_items:
			gradio.Markdown('*No source faces yet — drop photo(s) above.*')
			return

		with gradio.Row():
			for index, source_item in enumerate(source_items):
				with gradio.Column(min_width = 116):
					gradio.Image(
						value = source_item.get('crop'),
						height = 116,
						show_label = False,
						interactive = False,
						show_download_button = False,
						show_fullscreen_button = False,
						elem_classes = 'm2m-face-card'
					)
					remove_button = gradio.Button(value = '✕ Source ' + str(index + 1), size = 'sm')
					remove_button.click(remove_source_at(index), inputs = SOURCE_FACES_STATE, outputs = SOURCE_FACES_STATE)


def render_target_matcher() -> None:
	@gradio.render(inputs = [ SOURCE_FACES_STATE, TARGET_FACES_STATE ])
	def render_matcher(source_items : List[Dict[str, Any]], target_items : List[Dict[str, Any]]) -> None:
		if not target_items:
			gradio.Markdown('*No target faces yet — drop a target photo above.*')
			return

		source_choices = [ ('— keep original —', KEEP_ORIGINAL) ] + [ ('Source ' + str(index + 1), index) for index in range(len(source_items)) ]

		with gradio.Row():
			for target_index, target_item in enumerate(target_items):
				selected_source = target_item.get('source_index', KEEP_ORIGINAL)
				if selected_source >= len(source_items):
					selected_source = KEEP_ORIGINAL

				with gradio.Column(min_width = 150):
					gradio.Image(
						value = target_item.get('crop'),
						height = 116,
						show_label = False,
						interactive = False,
						show_download_button = False,
						show_fullscreen_button = False,
						elem_classes = 'm2m-face-card'
					)
					source_dropdown = gradio.Dropdown(
						label = 'Target ' + str(target_index + 1) + ' → replace with',
						choices = source_choices,
						value = selected_source,
						container = True
					)
					source_dropdown.change(assign_source_at(target_index), inputs = [ source_dropdown, TARGET_FACES_STATE ], outputs = TARGET_FACES_STATE)


def listen() -> None:
	SOURCE_FILE.change(detect_source_faces, inputs = SOURCE_FILE, outputs = [ SOURCE_FACES_STATE, STATUS_MARKDOWN ])
	TARGET_FILE.change(detect_target_faces, inputs = TARGET_FILE, outputs = [ TARGET_FACES_STATE, TARGET_FRAME_STATE, STATUS_MARKDOWN ])
	SWAP_BUTTON.click(swap_matched_faces, inputs = [ SOURCE_FACES_STATE, TARGET_FACES_STATE, TARGET_FRAME_STATE, SWAP_AREA_RADIO, ENHANCE_CHECKBOX ], outputs = [ RESULT_IMAGE, STATUS_MARKDOWN ])
	CLEAR_BUTTON.click(clear, outputs = [ SOURCE_FILE, TARGET_FILE, RESULT_IMAGE, STATUS_MARKDOWN, SOURCE_FACES_STATE, TARGET_FACES_STATE, TARGET_FRAME_STATE ])


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
	return numpy.ascontiguousarray(crop_vision_frame[:, :, ::-1])


def detect_faces_in_frame(vision_frame : VisionFrame) -> List[Dict[str, Any]]:
	clear_static_faces()
	faces = sort_faces_by_order(get_many_faces([ vision_frame ]), 'left-right')
	return [ { 'face': face, 'crop': crop_face(vision_frame, face) } for face in faces ]


def detect_source_faces(files : Optional[List[File]]) -> Tuple[List[Dict[str, Any]], gradio.Markdown]:
	source_paths = filter_image_paths([ file.name for file in files ]) if files else []

	if not source_paths:
		return [], gradio.Markdown(value = 'Add source faces and a target photo to begin.')

	source_items = []
	for source_path in source_paths:
		source_vision_frame = read_static_image(source_path)
		for source_item in detect_faces_in_frame(source_vision_frame):
			source_item['face'] = get_average_face([ source_item['face'] ])
			source_items.append(source_item)

	if not source_items:
		return [], gradio.Markdown(value = '❌  No face found in the source photos. Use clearer portraits.')
	return source_items, gradio.Markdown(value = '✅  Found ' + str(len(source_items)) + ' source face' + ('s' if len(source_items) != 1 else '') + '. Now match them to the target faces below.')


def detect_target_faces(target_path : Optional[str]) -> Tuple[List[Dict[str, Any]], Optional[VisionFrame], gradio.Markdown]:
	if not is_image(target_path):
		return [], None, gradio.Markdown(value = 'Add source faces and a target photo to begin.')

	target_vision_frame = read_static_image(target_path)
	target_items = detect_faces_in_frame(target_vision_frame)

	# auto-match each target face to the source at the same position
	for target_index, target_item in enumerate(target_items):
		target_item['source_index'] = target_index

	if not target_items:
		return [], None, gradio.Markdown(value = '❌  No face found in the target photo.')
	return target_items, target_vision_frame, gradio.Markdown(value = '✅  Found ' + str(len(target_items)) + ' target face' + ('s' if len(target_items) != 1 else '') + '. Pick a source for each one below.')


def remove_source_at(target_index : int) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
	def remove(source_items : List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		if source_items and 0 <= target_index < len(source_items):
			return [ source_item for index, source_item in enumerate(source_items) if index != target_index ]
		return source_items
	return remove


def assign_source_at(target_index : int) -> Callable[[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
	def assign(source_index : int, target_items : List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		target_items = list(target_items)
		if 0 <= target_index < len(target_items):
			updated_item = dict(target_items[target_index])
			updated_item['source_index'] = source_index
			target_items[target_index] = updated_item
		return target_items
	return assign


def apply_best_settings(swap_area : str, do_enhance : bool) -> None:
	area_options = SWAP_AREA_SET.get(swap_area, SWAP_AREA_SET.get('face'))
	state_manager.set_item('face_swapper_model', area_options.get('model'))
	state_manager.set_item('face_swapper_pixel_boost', area_options.get('pixel_boost'))
	# 'wide' uses box only so the swap reaches the forehead/jaw without occlusion
	# carving it back to the inner face; 'face' adds occlusion for clean overlaps.
	state_manager.set_item('face_mask_types', area_options.get('mask_types'))
	state_manager.set_item('face_mask_padding', [ 0, 0, 0, 0 ])

	if do_enhance:
		state_manager.set_item('face_enhancer_model', FACE_ENHANCER_MODEL)


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


def collect_pairs(source_items : List[Dict[str, Any]], target_items : List[Dict[str, Any]]) -> List[Tuple[int, int]]:
	pairs = []

	for target_index, target_item in enumerate(target_items):
		source_index = target_item.get('source_index', KEEP_ORIGINAL)
		if source_index is not None and 0 <= source_index < len(source_items):
			pairs.append((source_index, target_index))
	return pairs


def swap_matched_faces(source_items : List[Dict[str, Any]], target_items : List[Dict[str, Any]], target_vision_frame : Optional[VisionFrame], swap_area : str, do_enhance : bool):
	if not source_items:
		yield None, '⚠️  Add at least one source face.'
		return
	if not target_items or target_vision_frame is None:
		yield None, '⚠️  Add a target photo with at least one face.'
		return

	pairs = collect_pairs(source_items, target_items)
	if not pairs:
		yield None, '⚠️  Pick a source for at least one target face.'
		return

	yield None, '⏳  Preparing models (first run downloads them, please wait)…'
	apply_best_settings(swap_area, do_enhance)
	if not prepare_models(do_enhance):
		yield None, '❌  Could not prepare the required models. Check your connection and try again.'
		return

	result_vision_frame = target_vision_frame.copy()
	for order, (source_index, target_index) in enumerate(pairs):
		yield None, '✨  Swapping face ' + str(order + 1) + ' of ' + str(len(pairs)) + '…'
		result_vision_frame = face_swapper.swap_face(source_items[source_index]['face'], target_items[target_index]['face'], result_vision_frame)

	if do_enhance:
		yield None, '🎨  Enhancing faces…'
		result_vision_frame = enhance_faces(result_vision_frame)

	output_path = resolve_output_path()
	write_image(output_path, result_vision_frame)
	clear_static_faces()

	yield output_path, '✅  Swapped ' + str(len(pairs)) + ' face' + ('s' if len(pairs) != 1 else '') + '.'


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
		gradio.Image(value = None),
		gradio.Markdown(value = 'Add source faces and a target photo to begin.'),
		[],
		[],
		None
	)
