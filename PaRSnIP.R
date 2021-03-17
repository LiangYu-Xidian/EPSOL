#==================================================
#==================================================
#==================================================
# Libraries
library( bio3d )
library( stringr )
library( Interpol )
library( zoo )
library( data.table )

setwd('.')

#==================================================
#==================================================
#==================================================
# Functions

#==================================================
# Calculate features for test sequence
PaRSnIP.calc.features.test <- function( vec.seq,
                                        input.ss,
                                        input.ss8,
                                        input.acc20,
                                        AA = unlist( strsplit("ACDEFGHIKLMNPQRSTVWY",split = "" ) ),
                                        SS.3 = unlist( strsplit("CEH",split = "" ) ),
                                        SS.8 = unlist( strsplit("BCEGHIST",split = "" ) )
                                        )
{
    #==================================================
    # Preprocess sequence
    # Step 1: Remove all inserts ("-")
    vec.seq <- vec.seq[ vec.seq != "-" ]
    # Step 2: Convert all non-standard amino acids to X
    vec.seq[ !( vec.seq %in% AA ) ] <- "X"
    
    #==================================================
    # Sequence length
    p <- length( vec.seq )
    var.log.seq.len <- log( p )
    
    #==================================================
    # Calculate molecular weight
    # df.mw <- data.frame( read.csv( "DT_MW.csv" ) )
    df.mw <- data.frame( cbind( c( "A",
                                   "R",
                                   "N",
                                   "D",
                                   "C",
                                   "E",
                                   "Q",
                                   "G",
                                   "H",
                                   "I",
                                   "L",
                                   "K",
                                   "M",
                                   "F",
                                   "P",
                                   "S",
                                   "T",
                                   "W",
                                   "Y",
                                   "V" ),
                                c( 89.1,
                                   174.2,
                                   132.1,
                                   133.1,
                                   121.2,
                                   147.1,
                                   146.2,
                                   75.1,
                                   155.2,
                                   131.2,
                                   131.2,
                                   146.2,
                                   149.2,
                                   165.2,
                                   115.1,
                                   105.1,
                                   119.1,
                                   204.2,
                                   181.2,
                                   117.1 ) ) )
    colnames( df.mw ) <- c( "AA",
                            "MW" )
    df.mw$AA <- as.vector( df.mw$AA )
    df.mw$MW <- as.numeric( as.vector( df.mw$MW ) )
    
    
    vec.mw <- NULL
    for( i in 1:length( vec.seq ) )
    {
        if( nrow( df.mw[ df.mw$AA == vec.seq[ i ], ] ) > 0 )
        {
            vec.mw <- c( vec.mw,
                         df.mw[ df.mw$AA == vec.seq[ i ], ]$MW )
        }
    }
    # var.mw <- sum( vec.mw )
    var.mw <- log( sum( vec.mw ) )
    
    #==================================================
    # Frequency turn-forming residues
    vec.tfr <- 0
    for( i in 1:length( vec.seq ) )
    {
        if( vec.seq[ i ] %in% c( "N", "G", "P", "S" ) )
        {
            vec.tfr <- vec.tfr + 1  
        }
    }
    var.tfr <- vec.tfr / length( vec.seq )
    
    #==================================================
    # Calculate GRAVY index
    var.seq <- paste( vec.seq,
                      collapse = "" )
    var.gravy <- sum( unlist( Interpol::AAdescriptor( var.seq ) ) ) / length( vec.seq )
    
    #==================================================
    # Alipathic index
    vec.ali <- rep( 0,
                    4 )
    vec.ali[ 1 ] <- sum( vec.seq == "A" )
    vec.ali[ 2 ] <- sum( vec.seq == "V" )
    vec.ali[ 3 ] <- sum( vec.seq == "I" )
    vec.ali[ 4 ] <- sum( vec.seq == "L" )
    
    var.ali <- ( vec.ali[ 1 ] + 2.9 * vec.ali[ 2 ] + 3.9 * vec.ali[ 3 ] + 3.9 * vec.ali[ 4 ] ) / p
    
    #==================================================
    # Absolute charge
    vec.ch <- rep( 0,
                   4 )
    vec.ch[ 1 ] <- sum( vec.seq == "R" )
    vec.ch[ 2 ] <- sum( vec.seq == "K" )
    vec.ch[ 3 ] <- sum( vec.seq == "D" )
    vec.ch[ 4 ] <- sum( vec.seq == "E" )
    
    var.ch <- abs( ( ( vec.ch[ 1 ] + vec.ch[ 2 ] - vec.ch[ 3 ] - vec.ch[ 4 ] ) / p ) - 0.03 )
    
    #==================================================
    #==================================================
    #==================================================
    # SCRATCH features
    
    #==================================================
    # 3-state secondary structure classification

    vec.ss <- as.vector( input.ss )
    vec.ss.freq <- NULL
    for( i in 1:length( SS.3 ) )
    {
        vec.ss.freq <- c( vec.ss.freq,
                          sum( vec.ss == SS.3[ i ] ) )
    }
    vec.ss.freq <- vec.ss.freq / p
    
    #==================================================
    # 8-state secondary structure classification

    vec.ss8 <- as.vector( input.ss8 )
    vec.ss8.freq <- NULL
    for( i in 1:length( SS.8 ) )
    {
        vec.ss8.freq <- c( vec.ss8.freq,
                           sum( vec.ss8 == SS.8[ i ] ) )
    }
    vec.ss8.freq <- vec.ss8.freq / p
    
    #==================================================
    # Solvent accessibility prediction at 0%-95% thresholds

    vec.acc.20 <- as.numeric( input.acc20 )
    vec.thresh <- seq( 0, 95, 5 )
    # vec.acc.20.final <- NULL
    # for( i in 1:length( vec.thresh ) )
    # {
    #   vec.acc.20.final <- c( vec.acc.20.final,
    #                          sum( vec.acc.20 == vec.thresh[ i ] ) / p )
    # }
    vec.acc.20.final <- sum( vec.acc.20 == vec.thresh[ 1 ] ) / p
    for( i in 2:length( vec.thresh ) )
    {
        vec.acc.20.final <- c( vec.acc.20.final,
                               sum( vec.acc.20 >= vec.thresh[ i ] ) / p )
    }
    
    #==================================================
    # Solvent accessibility prediction at 0%-95% thresholds coupled with average hydrophobicity
    ind.thresh <- which( vec.acc.20 == vec.thresh[ 1 ] )
    if( length( ind.thresh ) == 0 )
    {
        vec.rsa.hydro <- 0
    } else
    {
        vec.rsa.hydro <- ( sum( vec.acc.20 == vec.thresh[ 1 ] ) / p ) *  
            mean( unlist( Interpol::AAdescriptor( vec.seq[ ind.thresh ] ) ) )
    }
    for( i in 2:length( vec.thresh ) )
    {
        ind.thresh <- which( vec.acc.20 >= vec.thresh[ i ] )
        if( length( ind.thresh ) == 0 )
        {
            vec.rsa.hydro <- c( vec.rsa.hydro,
                                0 )
        } else
        {
            vec.rsa.hydro <- c( vec.rsa.hydro,
                                ( sum( vec.acc.20 >= vec.thresh[ i ] ) / p ) *
                                    mean( unlist( Interpol::AAdescriptor( vec.seq[ ind.thresh ] ) ) ) )  
        }
    }
    
    
    #==================================================
    #==================================================
    #==================================================
    # Return feature vector
    vec.features <- c( var.log.seq.len,
                       var.mw,
                       var.tfr,
                       var.gravy,
                       var.ali,
                       var.ch,
                       #vec.AA.freq,
                       #vec.dipep.freq,
                       #vec.tripep.freq,
                       vec.ss.freq,
                       vec.ss8.freq,
                       vec.acc.20.final,
                       vec.rsa.hydro )
    
    return( vec.features )
}


#==================================================
#==================================================
# Main PaRSnIP function

PaRSnIP <- function( file.test,
                     file_ss.path,
                     file_ss8.path,
                     file_acc20.path,
                     file.output )
{
    
    # Load test sequence in fasta format
    print( "==================================================" )
    print( "Load test sequence in fasta format" )
    
    aln <- read.fasta( file.test )
    aln.ali <- aln$ali

    ss = read.fasta( file_ss.path )
    ss.ali <- ss$ali
    ss8 = read.fasta( file_ss8.path )
    ss8.ali <- ss8$ali

    acc20_str <- list()
    idx<-1
    nrow<-1
    con <-file( file_acc20.path,"r")
    line = readLines(con,n=1)
    while( length(line) != 0 ) {
        # print(line)
        if(nrow%%2==0){
            acc20_str[[idx]]<-line
            idx<-idx+1
        }
        line=readLines(con,n=1)
        nrow<-nrow+1
    }
    close(con)

    df_features <- NULL

    for (i in 1:nrow(aln.ali))
    {
        #==================================================
        # Calculate features for test sequence
        print( "==================================================" )
        print( "Calculate features for test sequence" )
        
        for (j in 1:length(aln.ali[i,])) 
        {
            if (aln.ali[i,length(aln.ali[i,])]!="-")
            {
                break;
            }
            else if (aln.ali[i,length(aln.ali[i,])-j]!='-')
            {
                break;
            }
        }
        print(paste0("Break point is: ",j))
        if (j==1) { j=0 }
        new.seq <- aln.ali[i,(1:(length(aln.ali[i,])-j))];
        vec.seq <- paste(new.seq,collapse="");
        print(paste0("Sequence is: ",vec.seq));

        ss_seq <- ss.ali[i,(1:(length(aln.ali[i,])-j))];
        print(ss_seq)
        ss8_seq <- ss8.ali[i,(1:(length(aln.ali[i,])-j))];
        print(ss8_seq)
        acc20_seq <- strsplit(acc20_str[[i]]," ")[[1]]
        print(acc20_seq)

        dummy_class <- 0
        
        vec.features <- PaRSnIP.calc.features.test( new.seq,
                                                    ss_seq,
                                                    ss8_seq,
                                                    acc20_seq )
        vec.features <- c(vec.features,dummy_class)
        df_features <- rbind(df_features,vec.features)

    }
    df_features <- as.data.frame(df_features)
    
    # Save features
    file.features <- paste( "./",file.output,
                            "_src_bio",
                            sep = "" )
    write.table( df_features, file = file.features, row.names=F,col.names=F, quote = F, sep="\t")

    print( "==================================================" )
    print( paste0("Features file saved in ", file.features) )
    print( "==================================================" )

}



#==================================================
#==================================================
#==================================================
# Main

#==================================================
# Command line arguments
file.test <- commandArgs()[ 3 ]
file_ss.path <- commandArgs()[ 4 ]
file_ss8.path <- commandArgs()[ 5 ]
file_acc20.path <- commandArgs()[ 6 ]
file.output <- commandArgs()[ 7 ]

if( is.na( file.output ) )
{
    file.output <- "result.txt"
}


# Run PaRSnIP
PaRSnIP( file.test,
         file_ss.path,
         file_ss8.path,
         file_acc20.path,
         file.output)
