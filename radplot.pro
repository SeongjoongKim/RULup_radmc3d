pro make_rr,radius,nbin,rrplt,rrpx,unit,pxsize=pxsize, linear=linear, dist=dist

    if keyword_set(linear) then begin
        rrpx = findgen(nbin)*(radius/dist/abs(pxsize)/nbin)+radius/dist/abs(pxsize)/nbin/2.0
        rrplt = findgen(nbin)*(radius/nbin) + (radius/nbin/2.0)
        unit = '[AU]'
    endif else begin
        if keyword_set(pxsize) then begin
            rrpx = findgen(nbin)*(radius/abs(pxsize)/nbin)+radius/abs(pxsize)/nbin/2.0
            rrplt = findgen(nbin)*(radius/nbin)+(radius/nbin/2.0)
            unit = '[arcsec]'
        endif else begin
            rrpx = findgen(nbin)*(radius/nbin)+(radius/nbin/2.0)
            rrplt = rrpx
            unit = '[pixel]'
        endelse
    endelse
    return
end

function atan2,y,x
    ;val = atan(y,x)
    ;val[where(val lt 0.0)] = val[where(val lt 0.0)] + val[where(val lt 0.0)]*2.0*!DPI
    val = atan(y,x)+(y lt 0.?2*!DPI:0.)
    return,val
end

function nonuniq, array, idx
    s = size(array)
    if (s[0] eq 0) then return, 0
    if n_params() ge 2 then begin
        q = array[idx]
        indices = where(q eq shift(q,-1), count)
        if (count gt 0) then return, idx[indices] $
        else return, n_elements(q)-1
    endif else begin
        indices = where(array eq shift(array,-1), count)
        if (count gt 0) then return, indices $
            else return, n_elements(array)-1
    endelse
end

pro make_coord,cx,cy,naxis0,naxis1,xx,yy,xmap,ymap,rmap,tmap,pxsize=pxsize,linear=linear,dist=dist,inc=inc,pa=pa
    if not keyword_set(dist) then disk=140.0
    if not keyword_set(inc) then inc=0.0
    if not keyword_set(pa) then pa=0.0

    if not keyword_set(pxsize) then pxsize=[1.0,1.0]
    if keyword_set(linear) then begin
        xx = (findgen(naxis0)-cx)*pxsize[0]*dist
        yy = (findgen(naxis1)-cy)*pxsize[1]*dist
        xmap = xx # replicate(1.0d0,n_elements(yy)) / cos(inc*!DTOR)
        ymap = replicate(1.0d0,n_elements(xx)) # yy
        rmap = sqrt(xmap^2+ymap^2)
        rmap = rot(rmap,-1.0*pa,1.0,cx,cy,/interp,/pivot)
        ;tmap = (-1.0*atan2(ymap,xmap)+!DPI/2)*!RADEG
        tmap = rot((-1.0*atan(ymap,xmap)+!DPI)*!RADEG,-90.0,1.0,cx,cy,/interp,/pivot)
        tmap = rot(tmap,-1.0*pa,1.0,cx,cy,/interp,/pivot)
        ;mkhdr,th1,rmap
        ;writefits,'rmap.fits',rmap,th1
        ;writefits,'tmap.fits',tmap,th1
    endif else begin
        xx = (findgen(naxis0)-cx)*pxsize[0]
        yy = (findgen(naxis1)-cy)*pxsize[1]
        xmap = xx # replicate(1.0,n_elements(yy))
        ymap = replicate(1.0,n_elements(xx)) # yy
        rmap = sqrt(xmap^2+ymap^2)
        ;tmap = (-1.0*atan(ymap,xmap)+!DPI/2)*!RADEG
        tmap = rot((-1.0*atan(ymap,xmap)+!DPI)*!RADEG,-90.0,1.0,cx,cy,/interp,/pivot)
        tmap = rot(tmap,-1.0*pa,1.0,cx,cy,/interp,/pivot)
    endelse
    return
end

pro radplot, fname, cx, cy, radius, nbin, azrange=azrange, $
             log=log, $
             pxsize=pxsize, $
             linear=linear, $
             dist=dist, $
             inc=inc, $
             pa=pa, $
             disp_annulus=disp_annulus, $
             disp_frac=disp_frac, $
             outfile=outfile, $
             noise=noise, $
             error_map=error_map, $
             error_map_beam=error_map_beam, $
             stderr=stderr, $
             stderr_beam=stderr_beam,$
             ckvar=ckvar

    ; cx, cy - center pixels
    ; radius - maximum radius of annulus ring
    ; nbin - number of bins. radius will be divided into this number
    ; azrange - azimuthial range to average. source PA is set to 0. default is [0,360]
    ; log - plot in log scale
    ; linear, inc, pa, dist - plot in linear scale (AU). all params are required.
    ; disp_annulus - display annular rings
    if not keyword_set(azrange) then azrange=[0.0,360.0] else azrange=azrange*1d
    if not keyword_set(linear) then dist=0.0
    if not keyword_set(inc) then inc=0.0
    if not keyword_set(pa) then pa=0.0
    radius=radius*1d
    nbin=nbin*1d

    raw = reform(readfits(fname,hd))
    ;raw = raw[*,*,1]
    ;mkhdr,th,raw
    ;writefits,'raw.fits',raw,th

    naxis = fxpar(hd, 'NAXIS*')
    cdelt = fxpar(hd, 'CDELT*')
    if keyword_set(pxsize) then begin
        if (n_elements(cdelt) gt 1) then pxsize=cdelt*3600.0 else pxsize=[-pxsize,pxsize]
    endif

    make_rr,radius,nbin,rrplt,rrpx,unit,pxsize=pxsize,linear=linear,dist=dist
    width = abs(rrplt[1]-rrplt[0])

    make_coord,cx,cy,naxis[0],naxis[1],xx,yy,xmap,ymap,rmap,tmap,pxsize=pxsize,linear=linear,dist=dist,inc=inc,pa=pa

    ave=dblarr(n_elements(rrplt))
    std=dblarr(n_elements(rrplt))
    thermal_std=dblarr(n_elements(rrplt))
    for i=0,n_elements(rrplt)-1 do begin
        ;print,where(rmap lt rrplt[i]+width/2.0 AND rmap ge rrplt[i]-width/2.0)
        ;print,where(tmap lt azrange[1] AND tmap ge azrange[0])
        rind = where(rmap lt rrplt[i]+width/2.0 AND rmap ge rrplt[i]-width/2.0)
        if (azrange[0] gt azrange[1]) then begin
            tind = where(tmap lt azrange[1] OR tmap ge azrange[0])
        endif else begin
            tind = where(tmap lt azrange[1] AND tmap ge azrange[0])
        endelse
        ind_c = [rind,tind]
        ind = ind_c[nonuniq(ind_c,sort(ind_c))]
        num_ind = n_elements(ind)
        ;print,i,n_elements(ind)
        if keyword_set(error_map_beam) then begin
            beam_omega = !DPI*fxpar(hd,'BMAJ')*3.6D3*fxpar(hd,'BMIN')*3.6D3/4.0/alog(2) $
                         / abs(pxsize[0]*pxsize[1]) ; beam in px
            ;ave[i] = sqrt(total(raw[ind]^2))/sqrt(num_ind/beam_omega)
            ave[i] = mean(abs(raw[ind]),/nan)/sqrt(num_ind/beam_omega)
        endif else begin
            if keyword_set(error_map) then begin
                ;ave[i] = sqrt(total(raw[ind]^2))/num_ind
                ave[i] = mean(abs(raw[ind]),/nan)/sqrt(num_ind)
            endif else begin
                ave[i] = mean(raw[ind],/nan) 
            endelse
        endelse
        if keyword_set(stderr_beam) then begin
            beam_omega = !DPI*fxpar(hd,'BMAJ')*3.6D3*fxpar(hd,'BMIN')*3.6D3/4.0/alog(2) $
                         / abs(pxsize[0]*pxsize[1]) ; beam in px
            std[i] = stddev(raw[ind],/nan,/double)/sqrt(n_elements(raw[ind])/beam_omega)
        endif else begin
            if keyword_set(stderr) then begin
                std[i] = stddev(raw[ind],/nan,/double)/sqrt(n_elements(raw[ind]))
            endif else begin
                std[i] = stddev(raw[ind],/nan,/double)
            endelse
        endelse
        if keyword_set(noise) then begin
            beam_omega = !DPI*fxpar(hd,'BMAJ')*3.6D3*fxpar(hd,'BMIN')*3.6D3/4.0/alog(2) $
                         / abs(pxsize[0]*pxsize[1])
            thermal_std[i]=noise/sqrt(num_ind/beam_omega)
        endif else begin
            thermal_std[i]=0.0
        endelse
        if keyword_set(ckvar) then begin
            ckvarf=file_basename(fname.Replace('.fits',''))+'_R'+strtrim(string(rrplt[i],format='(F05.1)'),1)+'_az'+strtrim(string(azrange[0],format='(I03)'),1)+'-'+strtrim(string(azrange[1],format='(I03)'),1)+'.dat'
            openw,1,ckvarf
            printf,1,reform(raw[ind],[1,num_ind])
            close,1
        endif
    endfor

    if keyword_set(disp_annulus) then begin
        ; plot image and annulus
        if not keyword_set(disp_frac) then frac=0.9
        sigr=sigrange(raw,fraction=frac,range=range)
       ;---- for using plot function
        iwin=getwindows('display of annulus')
        if (iwin ne !NULL) then begin
            iwin.window.setcurrent
            iwin.window.erase
            ireplot=1
        endif else begin
            ireplot=0
        endelse
        im1 = image(reverse(sigr,1), reverse(xx), yy, $
                    xrange=[xx[0],xx[n_elements(xx)-1]], $
                    yrange=[yy[0],yy[n_elements(yy)-1]], $
                    axis_style=2, $
                    xtitle='dR.A. '+unit, $
                    ytitle='dDec. '+unit, $
                    window_title='display of annulus', $
                    current=ireplot)
        for i=0,n_elements(rrplt)-1 do begin
            e=ellipse(0, 0, '-b', /data, target=im1, $
                      major=rrplt[i]-width/2.0, $
                      minor=(rrplt[i]-width/2.0)*cos(inc*!DTOR), $
                      theta=-90.0-1.0*pa, $
                      fill_background=0)
            e=ellipse(0, 0, '-b', /data, target=im1, $
                      major=rrplt[i]+width/2.0, $
                      minor=(rrplt[i]+width/2.0)*cos(inc*!DTOR), $
                      theta=-90.0-1.0*pa, $
                      fill_background=0)
        endfor
        azpos_x = rrplt[n_elements(rrplt)-1]*[cos(-1.*(azrange[0]*!DTOR)+!DPI/2-pa*!DTOR),0,cos(-1.0*(azrange[1]*!DTOR)+!DPI/2-pa*!DTOR)]
        azpos_y = rrplt[n_elements(rrplt)-1]*[sin(-1.*(azrange[0]*!DTOR)+!DPI/2-pa*!DTOR),0,sin(-1.0*(azrange[1]*!DTOR)+!DPI/2-pa*!DTOR)]
        pline = polyline(azpos_x,azpos_y,connectivity=[3,0,1,2],color=!color.red,/data,target=im1)
       ;---- for using plot proceduure
;        window,1,xsize=naxis[0],ysize=naxis[1]
;        contour,alog10(sigr), xx, yy, $
;                xrange=[xx[0],xx[n_elements(xx)-1]], $
;                yrange=[yy[0],yy[n_elements(yy)-1]], $
;                xtitle='dR.A.', ytitle='dDec.'
;        for i=0,n_elements(rrplt)-1 do begin
;            tvellipse, rrplt[i]-width/2.0, $
;                       (rrplt[i]-width/2.0)*cos(inc*!DTOR), $
;                       0, 0, $
;                       -90.0-1.0*pa, $
;                       /data
;            tvellipse, rrplt[i]+width/2.0, $
;                       (rrplt[i]+width/2.0)*cos(inc*!DTOR), $
;                       0, 0, $
;                       -90.0-1.0*pa, $
;                       /data
;        endfor
    endif

    ; plot radial distribution
    ;pwin=getwindows('radial plot')
    ;if (pwin ne !NULL) then begin
    ;    pwin.window.setcurrent
    ;    pwin.window.erase
    ;    preplot=1
    ;endif else begin
    ;    preplot=0
    ;endelse
    ;if keyword_set(log) then begin
    ;    xlog=1
    ;    ylog=1
    ;endif

    if keyword_set(outfile) then begin
        openw,1,outfile
        printf,1,[transpose(rrplt),transpose(ave),transpose(std),transpose(thermal_std),transpose(sqrt(std^2+thermal_std^2))]
        close,1
    endif

    ; remove missing data
    ;ind=finite(ave) AND finite(std)
    ind=finite(ave)
    rrplt = rrplt[where(ind eq 1)]
    ave = ave[where(ind eq 1)]
    std = std[where(ind eq 1)]
    if keyword_set(noise) then begin
        thermal_std = thermal_std[where(ind eq 1)]
        std = sqrt(std^2 + thermal_std^2)
    endif
    
    xrange=[0,max(rrplt)*1.2]
    yrange=[min(ave),max(ave)]
    p1=errorplot(rrplt, ave, std, "ko", $
                 /sym_filled, $
                 xlog=xlog, ylog=ylog, $
                 xrange=xrange, $
                 ;xstyle=0, $
                 ;yrange=yrange, $
                 ystyle=0, $
                 xtitle='R '+unit, $
                 ytitle='Intensity', $
                 window_title='radial plot', $
                 current=preplot)
end

