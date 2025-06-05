/*
	"Cheeseburger V2" by Xor (@XorDev)

	This is a revision of my original "Cheeseburger" shader: https://www.shadertoy.com/view/Wsf3D7


	License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
*/


//Anti-Aliasing level. Use 2, 3 or 4 for better quality.
#define AA 1.

vec3 dir = normalize(vec3(-4,1,3));

float soft(float a, float b, float n)
{
 	return log(exp(a*n)+exp(b*n))/n;   
}
vec3 hash(vec3 p)
{
 	return fract(sin(p*mat3(45.32,32.42,-41.55,65.35,-43.42,51.55,45.32,29.82,-41.45))*vec3(142.83,253.36,188.64));
}
vec3 value(vec3 p)
{
    vec3 f = floor(p);
    vec3 s = p-f;
    s *= s*(3.-s-s);
    const vec2 o = vec2(0,1);
    
    return mix(mix(mix(hash(f+o.xxx),hash(f+o.yxx),s.x),
        		   mix(hash(f+o.xyx),hash(f+o.yyx),s.x),s.y),
               mix(mix(hash(f+o.xxy),hash(f+o.yxy),s.x),
                   mix(hash(f+o.xyy),hash(f+o.yyy),s.x),s.y),s.z);
}
float worley(vec3 p)
{
    float d = 1.;
    for(int x = -1;x<=1;x++)
    for(int y = -1;y<=1;y++)
    for(int z = -1;z<=1;z++)
    {
    	vec3 f = floor(p+vec3(x,y,z));
        vec3 v = p-f-hash(f);
        d = soft(d,dot(v,v),-6.);
    }
    return d;
}
float seed(vec3 p)
{
    float d = 1.;
    for(int x = -1;x<=1;x++)
    for(int y = -1;y<=1;y++)
    {
    	vec3 f = floor(vec3(p.xy,0)+vec3(x,y,0));
        vec3 h = hash(f)*vec3(1,1,63);
        vec3 v = mat3(cos(h.z),sin(h.z),0,sin(h.z),-cos(h.z),0,0,0,1)*(p-f-h*.9)*vec3(1.7,1,0);
        d = min(d,dot(v,v)+step(9.,length(f+.6))+step(p.z,2.));
    }
    return max(.05-d,0.);
}
float cheese(vec3 p)
{
    p.z += -.27+.03*p.x*p.x+.1*soft(dot(p.xy,p.xy)-3.5,0.,10.);
 	return (length(max(abs(p)-vec3(1.6,1.6,0),0.))-.02)*.8;
}
float model(vec3 p)
{
    float d = length(p)-2.5;
    float m = soft(length(p.xy)-3.,pow(p.z-soft(d,0.,20.)*.7+1.1,2.)-.01,10.);
    //m = min(m,soft(p.z+1.2,-abs(mod(p.y+3.,4.)-2.),12.));
    if (d<.1)
    {
    	vec3 c = vec3(p.xy,max(p.z-.35,0.)*1.6);
    	float b = soft(length(c+.05*sin(c.yzx*2.))*.6-1.15,.41-abs(p.z+.15)-.02*c.x*c.x,40.);
        m = min(m,soft(b,-1.-p.z,20.));
    	m = min(m,soft(length(p.xy+.1*sin(c.yx*2.))-2.1,pow(p.z-.03+.03*p.x*p.x,2.)-.04,12.));
    	m = min(m,soft(length(p)-1.9,abs(p.z+.4-.03*p.y*p.y)-.1,80.));
    	m = min(m,cheese(p));
        vec3 r = value(p/dot(p,p)*vec3(14,14,1))-.5;
        vec3 l = p+vec3(0,0,.46)+vec3(0,0,length(p.xy)-1.8)*.3*cos(r.x*5.-r.y*5.);
    	m = min(m,soft(length(l)-2.1-.4*r.z,abs(l.z)-.02,28.)*.8);
        
        float s = .2*seed(p*5.);
        return m-s;
    }
	return min(d,m);
}
float bump(vec3 p)
{
    float b = .007*max(1.-1.5*abs(p.z),0.)*worley(p*18.);
          b += .001*(worley(p*30.)+.5*worley(p*70.))*step(-.99,p.z-max(length(p.xy)-2.,0.));
 	return model(p)+b*step(.01,cheese(p));
}
vec3 normal(vec3 p)
{
	vec3 n = vec3(-1,1,0)*.001;
    return normalize(bump(p+n.xyy)*n.xyy+bump(p+n.yxy)*n.yxy+bump(p+n.yyx)*n.yyx+bump(p+n.xxx)*n.xxx);
}

float light(vec3 p,vec3 n)
{
    float d = pow(2.-1.*dot(n,dir),-1.);
    d /= (1.-min(dot(dir,p),0.)*exp(.6-length(cross(dir,p))));
    
    float a = .5+.5*model(p+n*.05)/.05;
    a *= .6+.4*model(p+n*.1)/.1;
    a *= .7+.3*model(p+n*.2)/.2;
    return  d*a;
}
vec3 Tex(vec3 p,vec3 r)
{
    vec3 Rad = value(p/dot(p,p)*vec3(14,14,.2))-.5;
    vec3 l = p+vec3(0,0,.46)+vec3(0,0,length(p.xy)-1.8)*.3*cos(Rad.x*5.-Rad.y*5.);
   
    float t = max(length(p)-1.9,abs(p.z+.4-.03*p.y*p.y)-.1);
    
    vec3 n = normal(p);
    float w = worley(p*11.)*(.05+.95*smoothstep(.7,.4,abs(p.z+.04)))*(abs(n.z)*.7+.3);
    
 	float m = abs(p.z-.03+.03*p.x*p.x)-.3;
    float d = light(p,n);
    float s = max(dot(reflect(r,n),dir),0.);
    float f = length(p.xy);
    
    vec3 b = vec3(1,.635,.32)+.8*smoothstep(.9,.4,abs(p.z+.1)+.4*w)+.4*s*s+vec3(.7+.3*s*s*s)*seed(p*5.)/.1;
    vec3 c = mix(b,vec3(.7,.4,.2)-.6*w+.6*s*s*s,step(m,-.08));
    c = mix(c,vec3(.5,.8,.2)-.4*w+.6*s,step(abs(l.z),.04));
    c = mix(c,vec3(1,.6,.1)+.6*s*s,step(cheese(p),.002));
    c = mix(c,vec3(1,.2,.1)+s*s,step(t,.001));
    c = mix(c,vec3(1.5+.3*smoothstep(.03,.04,abs(f-2.6)))+s*s*s*s,step(p.z-max(f-2.,0.),-.99));
    c = mix(c,pow(texture(iChannel0,p.xy/3.).rgb,vec3(1,.9,1.2))*.7+.1,step(p.z,-1.18));
    
    return c*d;
}
//void mainImage( out vec4 Color, in vec2 Coord)
//{
//    vec2 a = vec2(iTime*.1,.2*cos(iTime*.1)+1.8);
//    vec3 x = vec3(cos(a.x)*sin(a.y),sin(a.x)*sin(a.y),cos(a.y)),
//         y = normalize(cross(x,vec3(0,0,-1))),
//    	 z = normalize(cross(x,y));
//    
//    mat3 m = mat3(x,y,z);
//    
//    vec3 c = vec3(0);
//    
//    for(float i = 0.;i<AA;i++)
//    for(float j = 0.;j<AA;j++)
//    {
//        vec4 p = vec4(m*vec3(-6,0,2.*cos(a.y)+.2),0);
//        vec3 r = m*vec3(1,(vec2(i,j)/AA+Coord-.5*iResolution.xy)/iResolution.y);
//        for(int I = 0;I<800;I++)
//        {
//            float s = model(p.xyz);
//            p += vec4(r,1)*s;
//            if ((p.w>20.) || (s<.001)) break;
//        }
//    	c += mix(Tex(p.xyz,r),vec3(.9)+.1*dot(dir,r),smoothstep(.5,1.,length(p.xyz)/10.));
//    }
//    Color = vec4(c/AA/AA,1);
//}
