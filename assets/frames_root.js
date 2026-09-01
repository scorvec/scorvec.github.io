// The ONE place the animation-frame host is set. Manifests stay on main and
// are fetched same-origin; only the <img> requests go here, and images never
// need CORS. Keys mirror repo paths (assets/<product>/anim/<dir>/<frame>.webp),
// so this is a prefix and nothing else - see scripts/lib/frames_store.py.
//
// Cutover: change this line to the bucket's public URL (trailing slash) once
// the frames-migrate workflow has copied the branch across.
window.FRAME_ROOT = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/frames/";
