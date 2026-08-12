import math
import numpy as np
import moderngl
from PIL import Image
def create_grid_coords(N):
    W = math.ceil(math.sqrt(N))
    # Create indices 0, 1, 2, ..., N-1
    indices = np.arange(N)
    # Convert to x, y coordinates using row-major ordering
    y,x = indices % W,indices // W
    # Stack into N x 2 array
    return np.column_stack([x, y])/W

def read_shader(path:str):
    result=""
    with open(path, 'r') as file:
        
        result= file.read()
    return result
def prepend_defines(shader_source, entity_count):
    content_to_insert=f"#define ENTITY_COUNT {entity_count}\n"
    return shader_prepend(shader_source,content_to_insert)
def shader_prepend(shader_source, content_to_insert):
    first_newline = shader_source.find('\n')
    return shader_source[:first_newline+1] + content_to_insert + shader_source[first_newline+1:]

MUTED_TRYSET_WARNINGS={}
def tryset(program:moderngl.Program,uniform,value):
    """
    Gracefully handle a uniform that doesn't appear in program.
    Uniforms are frequently optimized out if they are not used in the current version of the shader.
    """
    if uniform in program:
        program[uniform]=value
    else:
        global MUTED_TRYSET_WARNINGS
        if uniform not in MUTED_TRYSET_WARNINGS:
            MUTED_TRYSET_WARNINGS[uniform]=0
        MUTED_TRYSET_WARNINGS[uniform]+=1
        if MUTED_TRYSET_WARNINGS[uniform]<10:
            print('Warning: ',uniform,' not present in ',program)

def tryset_mat3(program: moderngl.Program, uniform, mat):
    """Upload a 3x3 matrix uniform (column-major float32).

    Gracefully handles optimized-away uniforms like tryset.
    """
    if uniform in program:
        program[uniform].write(mat.astype("f4").T.tobytes())
    else:
        global MUTED_TRYSET_WARNINGS
        if uniform not in MUTED_TRYSET_WARNINGS:
            MUTED_TRYSET_WARNINGS[uniform] = 0
        MUTED_TRYSET_WARNINGS[uniform] += 1
        if MUTED_TRYSET_WARNINGS[uniform] < 10:
            print('Warning: ', uniform, ' not present in ', program)

def readback_rule(rule_buffer, rule_index):
    """
    Read back a single Rule from the buffer at the specified index.
    
    Structure:
    - FourierCenter: vec4 frequency + vec4 amplitude = 8 floats = 32 bytes
    - Rule: 10 RbfCenters = 10 * 32 = 320 bytes
    """
    
    # Calculate the byte offset for the specific rule
    rule_size_bytes = 320  # 10 centers * 32 bytes per center
    offset = rule_index * rule_size_bytes
    
    # Read the specific rule from the buffer
    rule_bytes = rule_buffer.read(size=rule_size_bytes, offset=offset)
    
    # Convert bytes to numpy array
    # Each Rule contains 80 floats (10 centers * 8 floats per center)
    rule_data = np.frombuffer(rule_bytes, dtype=np.float32)
    
    # Reshape to [10 centers, 8 floats per center]
    rule_reshaped = rule_data.reshape(10, 8)
    
    return rule_reshaped
def set_rule_uniform(program, rule_data):
    """
    Set a Rule as a uniform in the shader program.

    Args:
        example_prog: ModernGL program object
        rule_data: numpy array of shape (10, 8) containing the rule data
    """

    # Method 1: Set individual FourierCenter uniforms
    for i in range(10):
        center_data = rule_data[i]
        frequency = center_data[:4]      # First 4 floats are frequency
        amplitude = center_data[4:]      # Last 4 floats are amplitude

        # Set uniforms (assuming uniform names like target_rule.centers[0].frequency, etc.)
        try:
            program[f'target_rule.centers[{i}].frequency'] = tuple(frequency)
            program[f'target_rule.centers[{i}].amplitude'] = tuple(amplitude)
        except Exception:
            print('failed rule uniforms')

def load_image_as_texture(ctx, image_path):
    """
    Load an arbitrary image file and convert it to a ModernGL RGBA texture.
    
    Args:
        ctx: ModernGL context
        image_path: Path to the image file (JPEG, PNG, etc.)
    
    Returns:
        moderngl.Texture: RGBA texture object
    """
    # Load and convert image to RGBA
    img = Image.open(image_path)
    img = img.convert('RGBA')  # Ensure RGBA format
    
    # Get image dimensions
    width, height = img.size
    
    # Get raw pixel data as bytes
    # PIL uses top-left origin, ModernGL uses bottom-left, so flip vertically
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    pixel_data = img.tobytes()
    
    # Create ModernGL texture
    texture = ctx.texture((width, height), 4, pixel_data)
    
    return texture