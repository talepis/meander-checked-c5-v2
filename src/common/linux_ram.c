/* Fresh Linux RSS, MemAvailable and cgroup-v2 snapshot, with reusable FDs.
   No sampled/cached RAM values. Only OS queries; no numerical kernels here. */
#define _GNU_SOURCE
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
static int fm=-1,fr=-1,fl[64],fu[64],fs[64],ng=0;
static uint64_t pages=0;
static int readfresh(int fd,char *buf,size_t size){
    ssize_t n=pread(fd,buf,size-1,0);
    if(n<=0 || (size_t)n==size-1){errno=n<0?errno:EIO;return -1;}
    buf[n]=0;return 0;
}
static int number(const char *text,uint64_t *value){
    char *end;errno=0;unsigned long long x=strtoull(text,&end,10);
    if(errno || end==text)return -1;
    *value=x;return 0;
}
static uint64_t field(const char *text,const char *key){
    size_t n=strlen(key);const char *p=text;
    while(*p){if(!strncmp(p,key,n) && (p[n]==' '||p[n]==':'||p[n]=='\t')){while(p[n]==' '||p[n]==':'||p[n]=='\t')n++;return strtoull(p+n,NULL,10);}p=strchr(p,'\n');if(!p)break;p++;}return 0;
}
int ram_init(const char **dirs,int count){
    if(count<0||count>64||fm!=-1)return EINVAL;
    fm=open("/proc/meminfo",O_RDONLY|O_CLOEXEC);fr=open("/proc/self/statm",O_RDONLY|O_CLOEXEC);
    if(fm<0||fr<0)return errno;
    long p=sysconf(_SC_PAGESIZE);if(p<=0)return EIO;pages=(uint64_t)p;
    for(int i=0;i<count;i++){
        char path[4096];
        if(snprintf(path,sizeof(path),"%s/memory.max",dirs[i])>=(int)sizeof(path))return ENAMETOOLONG;
        fl[i]=open(path,O_RDONLY|O_CLOEXEC);
        snprintf(path,sizeof(path),"%s/memory.current",dirs[i]);fu[i]=open(path,O_RDONLY|O_CLOEXEC);
        snprintf(path,sizeof(path),"%s/memory.stat",dirs[i]);fs[i]=open(path,O_RDONLY|O_CLOEXEC);
        if(fl[i]<0||fu[i]<0||fs[i]<0)return errno;
    }
    ng=count;return 0;
}
int ram_snapshot(uint64_t *out){
    char memory[8192],resident[256],small[128],stats[8192];uint64_t rss,available,total;
    if(fm<0)return EINVAL;
    if(readfresh(fm,memory,sizeof(memory))||readfresh(fr,resident,sizeof(resident)))return errno;
    char *p=strchr(resident,' ');if(!p||number(p,&rss))return EIO;
    rss*=pages;available=field(memory,"MemAvailable")*1024;total=field(memory,"MemTotal")*1024;
    if(!available||!total)return EIO;
    for(int i=0;i<ng;i++){
        uint64_t limit,used;
        if(readfresh(fl[i],small,sizeof(small)))return errno;
        if(!strncmp(small,"max",3))continue;
        if(number(small,&limit)||!limit)return EIO;
        if(limit>=(UINT64_C(1)<<60))continue;
        if(readfresh(fu[i],small,sizeof(small))||number(small,&used))return EIO;
        if(readfresh(fs[i],stats,sizeof(stats)))return errno;
        uint64_t inactive=field(stats,"inactive_file"),dirty=field(stats,"file_dirty"),wb=field(stats,"file_writeback");
        uint64_t clean=(__uint128_t)dirty+wb>=inactive?0:inactive-dirty-wb;
        __int128 capacity=(__int128)limit-used+clean;
        uint64_t avail=capacity<=0?0:capacity>UINT64_MAX?UINT64_MAX:(uint64_t)capacity;
        if(avail<available)available=avail;
        if(limit<total)total=limit;
    }
    out[0]=rss;out[1]=available;out[2]=total;return 0;
}
