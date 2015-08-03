#include "framework.h"
#include "tbb/tbb.h"
#include "basic_types.h"

using namespace prototyper;

framework frm;

void write_backbuffer( const vec2& pos, const vec4& color );
#define uniform

namespace shader_code
{
//////////////////////////////////////////////////
// Shader code from here
//////////////////////////////////////////////////

struct octree_node
{
  //first bit ... last bit
  //(1 << 31) ... (1 << 0)

  //1 bit max subdiv flag (is this the maximum resolution, or can we descend more?)
  //1 bit data type flag (is this a color node or a brick pointer node?)
  //30 bit child pointer (use 0...7 offset to address child nodes)
  u32 a;

  //if the data type flag is 0, this is a constant color node
  //  so RGBA8 color is stored here (alpha == 0 means empty!)
  //else
  //  the first 2 bits is unused, then a RGB10 brick pointer is stored
  u32 b;
};

//helper namespace to quickly access functions
//on the gpu we can't do classes, so that's why I did it this way...
namespace octree
{
  //http://www.volume-gfx.com/wp-content/uploads/2013/02/octreeNumbering.png
  enum octree_children_access
  {
    FAR_BOTTOM_LEFT = 0,
    FAR_BOTTOM_RIGHT,
    NEAR_BOTTOM_RIGHT,
    NEAR_BOTTOM_LEFT,
    FAR_TOP_LEFT,
    FAR_TOP_RIGHT,
    NEAR_TOP_RIGHT,
    NEAR_TOP_LEFT
  };

  vec3 octree_children_offset[] = 
  {
    vec3(0,0,1),
    vec3(1,0,1),
    vec3(1,0,0),
    vec3(0,0,0),
    vec3(0,1,1),
    vec3(1,1,1),
    vec3(1,1,0),
    vec3(0,1,0)
  };

  void init( octree_node& o )
  {
    o.a = o.b = 0;
  }
  
  //true if constant color
  //false if brick pointer
  bool get_data_type( const octree_node& o )
  {
    return (o.a >> 30) & 1;
  }

  void set_data_type( octree_node& o, bool s )
  {
    if( s )
    {
      o.a |= (1 << 30);
    }
    else
    {
      o.a &= ~(1 << 30);
    }
  }

  bool get_max_subdiv( const octree_node& o )
  {
    return o.a >> 31;
  }

  void set_max_subdiv( octree_node& o, bool s )
  {
    if( s )
    {
      o.a |= (1 << 31);
    }
    else
    {
      o.a &= ~(1 << 31);
    }
  }

  //the last 30 bits of addr is used!
  void set_child_pointer( octree_node& o, u32 addr )
  {
    o.a = (o.a & (3 << 30)) | 
        (addr & ~(3 << 30));
  }

  u32 get_child_pointer( const octree_node& o )
  {
    return (o.a & ~(3 << 30));
  }

  void set_constant_color( octree_node& o, const vec4& v )
  {
    o.b = 0;

    o.b |= int(v.x * 255) << 24;
    o.b |= int(v.y * 255) << 16;
    o.b |= int(v.z * 255) << 8;
    o.b |= int(v.w * 255);
  }

  vec4 get_constant_color( const octree_node& o )
  {
    vec4 r;
    r.x = (o.b >> 24);
    r.y = (o.b >> 16) & 0xff;
    r.z = (o.b >> 8 ) & 0xff;
    r.w = o.b & 0xff;

    return r * (1 / 255.0f);
  }

  //the last 30 bits of ptr is used!
  void set_brick_pointer( octree_node& o, const uvec3& ptr )
  {
    //the first 2 bits are unused, so we don't care about that...
    o.b = (ptr.x << 20) | ((ptr.y << 10) & ~(4095 << 20)) | (ptr.z & (1023));
  }

  uvec3 get_brick_pointer( const octree_node& o )
  {
    uvec3 ptr;

    ptr.x = o.b >> 20;
    ptr.y = (o.b >> 10) & 1023;
    ptr.z = o.b & 1023;

    return ptr;
  }
}

//uniforms
uniform float time; //time in seconds
uniform vec4 mouse; //(xy) mouse pos in pixels, (zw) clicks
uniform vec2 resolution; //screen resolution in pixels
uniform vec2 inv_resolution; // 1/resolution
uniform float aspect; // resoltion.y / resolution.x
uniform vec3 sky_color = vec3( 0.678, 0.874, 1.0 );
uniform octree_node node_pool[2000]; //root at 0 offset
uniform unsigned int next_free_address;
uniform vec3 world_size; //32^3  0...32
uniform vec3 voxel_size; //1^3

#ifndef EPSILON
#define EPSILON 0.001f
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38
#endif

#define INVALID (FLT_MAX)

//supersampling positions
uniform vec3 pos00 = vec3( 0.25, 0.25, 0 );
uniform vec3 pos10 = vec3( 0.75, 0.25, 0 );
uniform vec3 pos01 = vec3( 0.25, 0.75, 0 );
uniform vec3 pos11 = vec3( 0.75, 0.75, 0 );

struct ray
{
  vec3 pos, dir;
};

struct aabb
{
  vec3 min, max;
};

struct intersection
{
	float t;
};

struct camera
{
	vec3 pos, view, up, right;
};

uniform camera cam;

//in a shader this would be a simple prng
float rand()
{
  return frm.get_random_num( 0, 1 );
}

intersection intersect_aabb_ray( const aabb& ab, const ray& r )
{
  vec3 invR;

  // compute intersection of ray with all six bbox planes
#ifdef _DEBUG
  //in debug mode, pay attention to asserts
  for(int c = 0; c < 3; ++c )
  {
    if( mm::impl::is_eq(r.dir[c], 0) )
    {
      invR[c] = FLT_MAX;
    }
    else
    {
      invR[c] = 1.0f / r.dir[c];
    }
  }
#else
  //in release mode we dgaf about div by zero
  invR = 1.0f / r.dir;
#endif

	vec3 tbot = invR * (ab.min - r.pos);
	vec3 ttop = invR * (ab.max - r.pos);

	// re-order intersections to find smallest and largest on each axis
	vec3 tmin = min(ttop, tbot);
	vec3 tmax = max(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
	float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

	intersection i;
  i.t = smallest_tmax > largest_tmin ? (largest_tmin >= 0 ? largest_tmin : smallest_tmax) : INVALID;

	return i;
}

vec4 trace_recursive( const ray& rr, octree_node* ptr, float size, const aabb& ab, float dist )
{
  intersection i;
  ray r = rr;

  while(true)
  {
    if( octree::get_max_subdiv(*ptr) )
    {
      //if constant color
      if( octree::get_data_type(*ptr) )
      {
        return vec4(octree::get_constant_color(*ptr).xyz, dist);
      }
      else
      {
        //TODO no bricks yet...
        return -2;
      }
    }
    else
    { //we need to go deeper!
      octree_node* children = node_pool + octree::get_child_pointer(*ptr);
      size *= 0.5;

      int closest = -1;
      float closest_dist = FLT_MAX;
      vec4 result = -1;
      aabb tmp_ab;

      for(int c = 0; c < 8; ++c)
      {
        octree_node* child = children + c;

        if( octree::get_constant_color(*child).w <= 0 )
        {
          continue;
        }

        tmp_ab.min = ab.min + octree::octree_children_offset[c] * size;
        tmp_ab.max = tmp_ab.min + size;
        i = intersect_aabb_ray(tmp_ab, r);

        if( i.t != INVALID )
        {
          vec4 res = trace_recursive( r, child, size, tmp_ab, i.t );
          if( res.w > 0 && res.w < closest_dist )
          {
            closest = c;
            closest_dist = res.w;
            result = res;
          }
        }
      }

      if( closest < 0 )
      {
        return -1;
      }
      else
      {
        return result;
      }
    }   
  }
}

vec4 trace( const ray& rr )
{
  /**/
  ray r = rr;

  octree_node* ptr = node_pool;
  float size = world_size.x;
  aabb ab;
  ab.min = 0;
  ab.max = size;
  intersection i = intersect_aabb_ray(ab, r);
  float dist = i.t;

  if( i.t == INVALID )
  {
      return -2; //did not hit the volume
  }

  return trace_recursive( rr, ptr, size, ab, dist );
}

vec4 calculate_pixel( const vec3& pix_pos )
{
  //2x2 near plane, 90 degrees vertical fov
  vec3 plane_pos = pix_pos * vec3( inv_resolution, 1 ) * 2 - vec3( 1, 1, 0 );
  plane_pos.y *= aspect;

  ray r;
  r.pos = cam.pos + cam.view + cam.right * plane_pos.x + cam.up * plane_pos.y;
  r.dir = normalize(r.pos - cam.pos);

  vec4 res = trace( r );

  if( res.x > -1 ) //if found voxel
  {
    vec3 color = res.xyz;

    float sum = 0;

    /**/
    //place ray origin slightly above the surface
    vec3 int_point = r.pos + r.dir * ( res.w - 0.001f );

    //calculate AO (doesn't really look correct, but at least shows structural details, so don't care)
    int samples = 8;
    for( int c = 0; c < samples; ++c )
    {
      vec3 dir = normalize( vec3( rand(), rand(), rand() ) * 2 - 1 );
      float coeff = -sign( dot( dir, r.dir ) ); //sample along a hemisphere (would need surface normal, but this should be ok too)
      
      ray box_ray;
      box_ray.pos = int_point;
      box_ray.dir = dir * (coeff != 0 ? coeff : 1); //sign can be 0 
           
      vec4 pos = trace( box_ray );

      sum += pos.x < 0;
    }

    return sum / samples;
    /**/

    return 1;
  }
  else
  {
    //debugging colors
    if( res.x > -2 )
    {
      return 0; //no hit
    }
    else
    {
      return 0.5; //did not hit the volume
    }
  }
}

void main( const vec2& gl_FragCoord )
{
  //shader rand() would need this
  //seed += aspect * gl_FragCoord.x + gl_FragCoord.y * inv_resolution.y + time;
  vec2 uv = gl_FragCoord.xy * inv_resolution.xy;

  /**/
  //supersampling
  vec4 color = ( calculate_pixel( vec3( gl_FragCoord.xy, 0 ) + pos00 ) +
	calculate_pixel( vec3( gl_FragCoord.xy, 0 ) + pos01 ) +
	calculate_pixel( vec3( gl_FragCoord.xy, 0 ) + pos10 ) +
	calculate_pixel( vec3( gl_FragCoord.xy, 0 ) + pos11 ) ) * 0.25;
  /**/

  //vec4 color = calculate_pixel( vec3( gl_FragCoord.xy, 0 ) );

  write_backbuffer( gl_FragCoord, color );
}

//////////////////////////////////////////////////
}

//////////////////////////////////////////////////
// Code that runs the raytracer
//////////////////////////////////////////////////

uvec2 screen( 0 );
bool fullscreen = false;
bool silent = false;
string title = "Voxel rendering stuff";
vec4* pixels = 0;

//thread granularity stuff
struct startend
{
  int startx, endx, starty, endy;
};

//this basically divides the screen into 256 small parts
const unsigned threadw = 16;
const unsigned threadh = 16;
startend thread_startends[threadw * threadh];

void thread_func( startend s )
{
  for( int y = s.starty; y < s.endy; y++ )
  {
    for( int x = s.startx; x < s.endx; x++ )
    {
      shader_code::main( vec2( x, y ) );
    }
  }
}

void write_backbuffer( const vec2& pos, const vec4& color )
{
  assert( pos.x < screen.x && pos.y < screen.y && pos.x >= 0 && screen.y >= 0 );

  pixels[int(pos.y * screen.x + pos.x)] = color;
}

vec3 rotate_2d( const vec3& pp, float angle )
{
  vec3 p = pp;
	p.x = p.x * cos( angle ) - p.y * sin( angle );
	p.y = p.y * cos( angle ) + p.x * sin( angle );
	return p;
}

void calculate_ssaa_pos()
{
	float angle = atan( 0.5 );
	float stretch = sqrt( 5.0 ) * 0.5;

	shader_code::pos00 = rotate_2d( shader_code::pos00, angle );
	shader_code::pos01 = rotate_2d( shader_code::pos01, angle );
	shader_code::pos10 = rotate_2d( shader_code::pos10, angle );
	shader_code::pos11 = rotate_2d( shader_code::pos11, angle );

	shader_code::pos00 = ( shader_code::pos00 - vec3( 0.5, 0.5, 0 ) ) * stretch + vec3( 0.5, 0.5, 0 );
	shader_code::pos01 = ( shader_code::pos01 - vec3( 0.5, 0.5, 0 ) ) * stretch + vec3( 0.5, 0.5, 0 );
	shader_code::pos10 = ( shader_code::pos10 - vec3( 0.5, 0.5, 0 ) ) * stretch + vec3( 0.5, 0.5, 0 );
	shader_code::pos11 = ( shader_code::pos11 - vec3( 0.5, 0.5, 0 ) ) * stretch + vec3( 0.5, 0.5, 0 );
}

struct color
{
  unsigned char r, g, b, a;
};

shader_code::camera lookat( const vec3& eye, const vec3& lookat, const vec3& up )
{
	shader_code::camera c;
	c.view = normalize( lookat - eye );
	c.up = normalize( up );
	c.pos = eye;
	c.right = normalize( cross( c.view, c.up ) );
	c.up = normalize( cross( c.right, c.view ) );
	return c;
}

void recursive_fill_octree( shape* s, shader_code::octree_node* parent, float parent_size, const vec3& parent_pos )
{
  for(int c = 0; c < 8; ++c)
  {
    vec3 child_pos = parent_pos + shader_code::octree::octree_children_offset[c] * vec3(parent_size*0.5);
    aabb ab = aabb(child_pos + vec3(parent_size * 0.5 * 0.5), parent_size*0.5*0.5);
    
    shader_code::octree_node* child = shader_code::node_pool + shader_code::octree::get_child_pointer(*parent) + c;
    shader_code::octree::set_data_type(*child, true);

    if( s->is_intersecting(&ab) )
    {
      shader_code::octree::set_constant_color(*child, 1);

      if( parent_size * 0.5 > shader_code::voxel_size.x * 2 )
      {
        shader_code::octree::set_child_pointer(*child, shader_code::next_free_address);
        shader_code::next_free_address += 8;
        recursive_fill_octree(s, child, parent_size * 0.5, child_pos);
      }
      else
      {
        shader_code::octree::set_max_subdiv(*child, true);
      }
    }
    else
    {
      if( parent_size * 0.5 < shader_code::voxel_size.x * 2 )
          shader_code::octree::set_max_subdiv(*child, true);

      shader_code::octree::set_constant_color(*child, 0);
    }
  }
}

void fill_octree( shape* s, shader_code::octree_node* o )
{
  float size = shader_code::world_size.x;
  aabb ab = aabb(vec3(size*0.5), size*0.5);

  shader_code::next_free_address = 1;

  if( ab.is_intersecting(s) )
  {
    shader_code::octree::set_constant_color(*o, 1);
    shader_code::octree::set_data_type(*o, true);
    shader_code::octree::set_max_subdiv(*o, false);
    shader_code::octree::set_child_pointer(*o, shader_code::next_free_address);
    shader_code::next_free_address += 8;
    recursive_fill_octree(s, o, size, 0);
  }
}

int main( int argc, char** argv )
{
  shape::set_up_intersection();

  map<string, string> args;

  for( int c = 1; c < argc; ++c )
  {
    args[argv[c]] = c + 1 < argc ? argv[c + 1] : "";
    ++c;
  }

  cout << "Arguments: " << endl;
  for_each( args.begin(), args.end(), []( pair<string, string> p )
  {
    cout << p.first << " " << p.second << endl;
  } );

  /*
   * Process program arguments
   */

  stringstream ss;
  ss.str( args["--screenx"] );
  ss >> screen.x;
  ss.clear();
  ss.str( args["--screeny"] );
  ss >> screen.y;
  ss.clear();

  if( screen.x == 0 )
  {
    screen.x = 512;
  }

  if( screen.y == 0 )
  {
    screen.y = 512;
  }

  try
  {
    args.at( "--fullscreen" );
    fullscreen = true;
  }
  catch( ... ) {}

  try
  {
    args.at( "--help" );
    cout << title << ", written by Marton Tamas." << endl <<
         "Usage: --silent      //don't display FPS info in the terminal" << endl <<
         "       --screenx num //set screen width (default:1280)" << endl <<
         "       --screeny num //set screen height (default:720)" << endl <<
         "       --fullscreen  //set fullscreen, windowed by default" << endl <<
         "       --help        //display this information" << endl;
    return 0;
  }
  catch( ... ) {}

  try
  {
    args.at( "--silent" );
    silent = true;
  }
  catch( ... ) {}

  /*
   * Initialize the OpenGL context
   */

  frm.init( screen, title, fullscreen );

  //set opengl settings
  glEnable( GL_DEPTH_TEST );
  glDepthFunc( GL_LEQUAL );
  glFrontFace( GL_CCW );
  glEnable( GL_CULL_FACE );
  glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
  glClearDepth( 1.0f );

  frm.get_opengl_error();

  glViewport( 0, 0, screen.x, screen.y );

  /*
   * Set up the scene
   */

   shader_code::resolution = vec2( screen.x, screen.y );
   shader_code::aspect = shader_code::resolution.y / shader_code::resolution.x;
   shader_code::world_size = vec3( 32 );
   shader_code::voxel_size = vec3( 1 );

   pixels = new vec4[screen.x * screen.y];

  //set up the camera
	shader_code::cam = lookat( vec3( 10, 5, 24 ), vec3( 16, 16, 16 ), vec3( 0, 1, 0 ) );

  //calculate thread start/ends
  for( unsigned int x = 0; x < threadw; ++x )
  {
    for( unsigned int y = 0; y < threadh; ++y )
    {
      startend s;
      s.startx = ( screen.x / ( float )threadw ) * x;
      s.endx = ( screen.x / ( float )threadw ) * ( x + 1 );
      s.starty = ( screen.y / ( float )threadh ) * y;
      s.endy = ( screen.y / ( float )threadh ) * ( y + 1 );
      thread_startends[x * threadh + y] = s;
    }
  }

  //initialize thread building blocks
  tbb::task_scheduler_init();

  calculate_ssaa_pos();

  shader_code::inv_resolution = 1.0f / shader_code::resolution;

  memset( shader_code::node_pool, 0, sizeof(shader_code::node_pool) );
  sphere s = sphere( vec3( 16, 16, 16 ), 10 );
  fill_octree( &s, shader_code::node_pool );

  /*
   * Handle events
   */

  auto event_handler = [&]( const sf::Event & ev )
  {
    switch( ev.type )
    {
      case sf::Event::MouseMoved:
        {
          shader_code::mouse.x = ev.mouseMove.x;
          shader_code::mouse.y = screen.y - ev.mouseMove.y;
        }
      default:
        break;
    }
  };

  /*
   * Render
   */

  sf::Clock timer;
  timer.restart();

  sf::Clock movement_timer;
  movement_timer.restart();

  float move_amount = 10;
  float orig_move_amount = move_amount;

  vec3 movement_speed = vec3(0);

  cout << "Init finished, rendering starts..." << endl;

  frm.display( [&]
  {
    frm.handle_events( event_handler );

    float seconds = movement_timer.getElapsedTime().asMilliseconds() / 1000.0f;

    if( sf::Keyboard::isKeyPressed( sf::Keyboard::LShift ) || sf::Keyboard::isKeyPressed( sf::Keyboard::RShift ) )
    {
      move_amount = orig_move_amount * 3.0f;
    }
    else
    {
      move_amount = orig_move_amount;
    }

    if( seconds > 0.01667 )
    {
      //move camera
      if( sf::Keyboard::isKeyPressed( sf::Keyboard::A ) )
      {
        movement_speed.x -= move_amount;
      }

      if( sf::Keyboard::isKeyPressed( sf::Keyboard::D ) )
      {
        movement_speed.x += move_amount;
      }

      if( sf::Keyboard::isKeyPressed( sf::Keyboard::W ) )
      {
        movement_speed.y += move_amount;
      }

      if( sf::Keyboard::isKeyPressed( sf::Keyboard::S ) )
      {
        movement_speed.y -= move_amount;
      }

      if( sf::Keyboard::isKeyPressed( sf::Keyboard::Q ) )
      {
        movement_speed.z -= move_amount;
      }

      if( sf::Keyboard::isKeyPressed( sf::Keyboard::E ) )
      {
        movement_speed.z += move_amount;
      }

      {
        shader_code::cam.pos += vec3(0,0,-1) * movement_speed.y * seconds;
        shader_code::cam.pos += vec3(-1,0,0) * movement_speed.x * seconds;
        shader_code::cam.pos += vec3(0,1,0) * movement_speed.z * seconds;

        //set up the camera
	      shader_code::cam = lookat( shader_code::cam.pos, vec3( 16, 16, 16 ), vec3( 0, 1, 0 ) );
      }

      movement_speed *= 0.5;

      movement_timer.restart();
    }

    shader_code::time = timer.getElapsedTime().asMilliseconds() * 0.001f;

    //raytrace for each pixel in parallel
    /**/
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, threadw * threadh ),
    [ = ]( const tbb::blocked_range<size_t>& r )
    {
      for( size_t i = r.begin(); i != r.end(); ++i )
        thread_func( thread_startends[i] );
    });
    /**/

    /**
    //single threaded code for debugging
    for( int y = 0; y < screen.y; ++y )
      for( int x = 0; x < screen.x; ++x )
      {
        shader_code::main( vec2( x, y ) );
      }
    /**/

    glDrawPixels( screen.x, screen.y, GL_RGBA, GL_FLOAT, pixels );

    //cout << "frame" << endl;

    frm.get_opengl_error();
  }, silent );

  return 0;
}
