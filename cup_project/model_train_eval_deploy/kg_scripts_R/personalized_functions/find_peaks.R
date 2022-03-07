find_peaks <-function (x, m = 10^-3){
  #starting source code:
  #https://github.com/stas-g/findPeaks
  #https://stats.stackexchange.com/questions/22974/how-to-find-local-peaks-valleys-in-a-series-of-data
  # modified by: Enzo
  # a 'peak' is defined as a local maxima with m points either side of it being smaller than it.
  # hence, the bigger the parameter m, the more stringent is the peak funding procedure.
  #the function can also be used to find local minima of any sequential vector x via find_peaks(-x).
  shape <- diff(sign(diff(x, na.pad = FALSE)))
  #print(shape)
  pks <- sapply(which(shape < 0), FUN = function(i){
    #print("insapp")
    z <- i - m + 1
    #print(z)
    #print(i)
    z <- ifelse(z > 0, z, 1)
    #print(z)
    w <- i + m + 1
    #print("w")
    #print(w)
    w <- ifelse(w < length(x), w, length(x))
    #print(w)
    #print("endSapp")
    if(all(x[c(z : i, (i + 2) : w)] <= x[i + 1])) return(i + 1) else return(numeric(0))
  })
  pks <- unlist(pks)
  pks
}

#massimo tra i massimi
#  findpeaks(vect, sortstr = T, threshold = 10^-3)[1,2]  #,2 is the position  #1, if sorted is the maximum!

#primo tra i massimi
#   findpeaks(vect, sortstr = F, threshold = 10^-3)[1,2]  #,2 is the position  #1, if sorted is the maximum!


findpeaks <- function(x,nups = 1, ndowns = nups, zero = "0", peakpat = NULL, 
                      # peakpat = "[+]{2,}[0]*[-]{2,}", 
                      minpeakheight = -Inf, minpeakdistance = 1,
                      threshold = 0, npeaks = 0, sortstr = FALSE)
{
  #code https://rdrr.io/rforge/pracma/src/R/findpeaks.R
  stopifnot(is.vector(x, mode="numeric") || length(is.na(x)) == 0)
  if (! zero %in% c('0', '+', '-'))
    stop("Argument 'zero' can only be '0', '+', or '-'.")
  
  # transform x into a "+-+...-+-" character string
  xc <- paste(as.character(sign(diff(x))), collapse="")
  xc <- gsub("1", "+", gsub("-1", "-", xc))
  # transform '0' to zero
  if (zero != '0') xc <- gsub("0", zero, xc)
  
  # generate the peak pattern with no of ups and downs
  if (is.null(peakpat)) {
    peakpat <- sprintf("[+]{%d,}[-]{%d,}", nups, ndowns)
  }
  
  # generate and apply the peak pattern
  rc <- gregexpr(peakpat, xc)[[1]]
  if (rc[1] < 0) return(NULL)
  
  # get indices from regular expression parser
  x1 <- rc
  x2 <- rc + attr(rc, "match.length")
  attributes(x1) <- NULL
  attributes(x2) <- NULL
  
  # find index positions and maximum values
  n <- length(x1)
  xv <- xp <- numeric(n)
  for (i in 1:n) {
    xp[i] <- which.max(x[x1[i]:x2[i]]) + x1[i] - 1
    xv[i] <- x[xp[i]]
  }
  
  # eliminate peaks that are too low
  inds <- which(xv >= minpeakheight & xv - pmax(x[x1], x[x2]) >= threshold)
  
  # combine into a matrix format
  X <- cbind(xv[inds], xp[inds], x1[inds], x2[inds])
  
  # eliminate peaks that are near by
  if (minpeakdistance < 1)
    warning("Handling 'minpeakdistance < 1' is logically not possible.")
  
  # sort according to peak height
  if (sortstr || minpeakdistance > 1) {
    sl <- sort.list(X[, 1], na.last = NA, decreasing = TRUE)
    X <- X[sl, , drop = FALSE]
  }
  
  # return NULL if no peaks
  if (length(X) == 0) return(c())
  
  # find peaks sufficiently distant
  if (minpeakdistance > 1) {
    no_peaks <- nrow(X)
    badpeaks <- rep(FALSE, no_peaks)
    
    # eliminate peaks that are close to bigger peaks
    for (i in 1:no_peaks) {
      ipos <- X[i, 2]
      if (!badpeaks[i]) {
        dpos <- abs(ipos - X[, 2])
        badpeaks <- badpeaks | (dpos > 0 & dpos < minpeakdistance)
      }
    }
    # select the good peaks
    X <- X[!badpeaks, , drop = FALSE]
  }
  
  # Return only the first 'npeaks' peaks
  if (npeaks > 0 && npeaks < nrow(X)) {
    X <- X[1:npeaks, , drop = FALSE]
  }
  
  return(X)
}
