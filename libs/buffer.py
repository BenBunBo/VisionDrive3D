from .shader import *
import OpenGL.GL as GL
import cv2




class VAO(object):
    def __init__(self):

        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        GL.glBindVertexArray(0)
        self.vbo = {}
        self.ebo = None



    def add_vbo(self, location, data,
               ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None):
        self.activate() # VAO
        buffer_idx = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data, GL.GL_STATIC_DRAW)
        #location = GL.glGetAttribLocation(self.shader.render_idx, name)
        GL.glVertexAttribPointer(location, ncomponents, dtype, normalized, stride, offset)
        GL.glEnableVertexAttribArray(location)
        self.vbo[location] = buffer_idx
        self.deactivate() #VAO

    def add_ebo(self, indices):
        self.activate()
        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices, GL.GL_STATIC_DRAW)
        self.deactivate()


    def __del__(self):
        try:
            import sys
            if sys.meta_path is None:
                return          # interpreter is shutting down, skip cleanup
            GL.glDeleteVertexArrays(1, [self.vao])
            GL.glDeleteBuffers(1, list(self.vbo.values()))
            if self.ebo is not None:
                GL.glDeleteBuffers(1, [self.ebo])
        except Exception:
            pass               # swallow any remaining errors during shutdown

    def activate(self):
        GL.glBindVertexArray(self.vao)  # activated

    def deactivate(self):
        GL.glBindVertexArray(0)  # activated

class UManager(object):
    def __init__(self, shader):
        self.shader = shader
        self.textures = {}

    @staticmethod
    def load_texture(filename):
        texture = cv2.cvtColor(cv2.imread(filename, 1), cv2.COLOR_BGR2RGB)
        return texture

    def _get_texture_loc(self):
        if not bool(self.textures):
            return 0
        else:
            locs = list(self.textures.keys())
            locs.sort(reverse=True)
            ret_id = locs[0] + 1
            return ret_id

    """
    * first call to setup_texture: activate GL.GL_TEXTURE0
        > use GL.glUniform1i to associate the activated texture to the texture in shading program (see fragment shader)
    * second call to setup_texture: activate GL.GL_TEXTURE1
        > use GL.glUniform1i to associate the activated texture to the texture in shading program (see fragment shader)
    * second call to setup_texture: activate GL.GL_TEXTURE2
        > use GL.glUniform1i to associate the activated texture to the texture in shading program (see fragment shader)
    and so on
    
    """
    def setup_texture(self, sampler_name, image_file):
        rgb_image = UManager.load_texture(image_file)

        GL.glUseProgram(self.shader.render_idx) # must call before calling to GL.glUniform1i
        texture_idx = GL.glGenTextures(1)
        binding_loc = self._get_texture_loc()
        self.textures[binding_loc] = {}
        self.textures[binding_loc]["id"] = texture_idx
        self.textures[binding_loc]["name"] = sampler_name

        GL.glActiveTexture(GL.GL_TEXTURE0 + binding_loc) # activate texture GL.GL_TEXTURE0, GL.GL_TEXTURE1, ...
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_idx)
        GL.glUniform1i(GL.glGetUniformLocation(self.shader.render_idx, sampler_name),
                       binding_loc)

        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                        rgb_image.shape[1], rgb_image.shape[0],
                        0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb_image)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

    def upload_uniform_matrix4fv(self, matrix, name, transpose=True):
        GL.glUseProgram(self.shader.render_idx)
        location = GL.glGetUniformLocation(self.shader.render_idx, name)
        GL.glUniformMatrix4fv(location, 1, transpose, matrix)

    def upload_uniform_matrix3fv(self, matrix, name, transpose=False):
        GL.glUseProgram(self.shader.render_idx)
        location = GL.glGetUniformLocation(self.shader.render_idx, name)
        GL.glUniformMatrix3fv(location, 1, transpose, matrix)

    def upload_uniform_vector4fv(self, vector, name):
        GL.glUseProgram(self.shader.render_idx)
        location = GL.glGetUniformLocation(self.shader.render_idx, name)
        GL.glUniform4fv(location, 1, vector)

    def upload_uniform_vector3fv(self, vector, name):
        GL.glUseProgram(self.shader.render_idx)
        location = GL.glGetUniformLocation(self.shader.render_idx, name)
        GL.glUniform3fv(location, 1, vector)

    def upload_uniform_scalar1f(self, scalar, name):
        GL.glUseProgram(self.shader.render_idx)
        location = GL.glGetUniformLocation(self.shader.render_idx, name)
        GL.glUniform1f(location, scalar)

    def upload_uniform_scalar1i(self, scalar, name):
        GL.glUseProgram(self.shader.render_idx)
        location = GL.glGetUniformLocation(self.shader.render_idx, name)
        GL.glUniform1i(location, scalar)