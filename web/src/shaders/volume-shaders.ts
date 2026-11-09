/**
 * GLSL shaders for the ray-marched volume viewer.
 * Inlined as strings because Next.js/Turbopack doesn't support ?raw imports.
 * Adapted from shaders by TODO-ATTRIBUTION.
 */

export const VOLUME_VERT = /* glsl */ `
out vec3 vOrigin;
out vec3 vDirection;

void main() {
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vec4 cameraObj = inverse(modelMatrix) * vec4(cameraPosition, 1.0);
  vOrigin = cameraObj.xyz;
  vDirection = position - cameraObj.xyz;
  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

export const VOLUME_FRAG = /* glsl */ `
precision highp float;
precision highp sampler3D;

in vec3 vOrigin;
in vec3 vDirection;

uniform sampler3D uVolume;
uniform sampler2D uTransfer;
uniform vec2 uRange;
uniform float uSteps;
uniform vec3 uClipMin;
uniform vec3 uClipMax;
uniform vec2 uThresholdRange;
uniform vec2 uPan;
uniform vec2 uZoom;

uniform sampler3D uVolumeTheta;
uniform sampler3D uVolumeSalt;
uniform sampler3D uVolumeW;

uniform bool uHasTheta;
uniform bool uHasSalt;
uniform bool uHasW;

uniform vec2 uRangeTheta;
uniform vec2 uRangeSalt;
uniform vec2 uRangeW;

uniform vec2 uThresholdTheta;
uniform vec2 uThresholdSalt;
uniform vec2 uThresholdW;

out vec4 outColor;

vec2 hitBox(vec3 origin, vec3 dir) {
  vec3 invDir = 1.0 / dir;
  vec3 tbot = (uClipMin - origin) * invDir;
  vec3 ttop = (uClipMax - origin) * invDir;
  vec3 tmin = min(tbot, ttop);
  vec3 tmax = max(tbot, ttop);
  float tNear = max(max(tmin.x, tmin.y), tmin.z);
  float tFar  = min(min(tmax.x, tmax.y), tmax.z);
  return vec2(tNear, tFar);
}

void main() {
  vec3 dir = normalize(vDirection);
  vec2 t = hitBox(vOrigin, dir);
  if (t.x > t.y) discard;
  float tEntry = max(t.x, 0.0);
  float tExit = t.y;

  float steps = max(uSteps, 1.0);
  float dt = (tExit - tEntry) / steps;

  vec3 p = vOrigin + dir * tEntry;
  vec3 step = dir * dt;

  vec4 acc = vec4(0.0);
  float range = max(uRange.y - uRange.x, 1e-6);

  for (int i = 0; i < 512; i++) {
    if (float(i) >= steps) break;

    vec3 uvw;
    uvw.xy = (p.xy - uPan) * uZoom + 0.5;
    uvw.z = p.z + 0.5;

    if (uvw.x < 0.0 || uvw.x > 1.0 || uvw.y < 0.0 || uvw.y > 1.0) {
      p += step;
      continue;
    }

    bool passTheta = true;
    if (uHasTheta) {
      float s = texture(uVolumeTheta, uvw).r;
      float n = clamp((s - uRangeTheta.x) / max(uRangeTheta.y - uRangeTheta.x, 1e-6), 0.0, 1.0);
      passTheta = (n >= uThresholdTheta.x && n <= uThresholdTheta.y);
    }

    bool passSalt = true;
    if (uHasSalt) {
      float s = texture(uVolumeSalt, uvw).r;
      float n = clamp((s - uRangeSalt.x) / max(uRangeSalt.y - uRangeSalt.x, 1e-6), 0.0, 1.0);
      passSalt = (n >= uThresholdSalt.x && n <= uThresholdSalt.y);
    }

    bool passW = true;
    if (uHasW) {
      float s = texture(uVolumeW, uvw).r;
      float n = clamp((s - uRangeW.x) / max(uRangeW.y - uRangeW.x, 1e-6), 0.0, 1.0);
      passW = (n >= uThresholdW.x && n <= uThresholdW.y);
    }

    if (passTheta && passSalt && passW) {
      float scalar = texture(uVolume, uvw).r;
      float n = clamp((scalar - uRange.x) / range, 0.0, 1.0);

      if (n >= uThresholdRange.x && n <= uThresholdRange.y) {
        vec4 c = texture(uTransfer, vec2(n, 0.5));
        float opacityFactor = abs(scalar) < 1e-4 ? 0.0 : 1.0;
        float a = clamp(c.a * opacityFactor, 0.0, 1.0);
        acc.rgb += (1.0 - acc.a) * c.rgb * a;
        acc.a   += (1.0 - acc.a) * a;
        if (acc.a > 0.99) break;
      }
    }

    p += step;
  }

  if (acc.a <= 0.0) discard;
  outColor = vec4(acc.rgb, acc.a);
}
`;
