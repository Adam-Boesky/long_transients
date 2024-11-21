from mastcasjobs import MastCasJobs
from Extracting.utils import get_credentials

# Set up casjobs object
wsid, password = get_credentials('mast_login.txt')
jobs = MastCasJobs(context="PanSTARRS_DR2", userid=wsid, password=password, request_type='POST')
# query = "SELECT TOP 10 * FROM ObjectThin"
# jobid = jobs.submit(query, task_name="test")
# print('submitted!')
# jobs.monitor(jobid)

query = """
WITH
    g_band AS (
        SELECT
            o.objID,
            o.raMean,
            o.decMean,
            m.gKronMag,
            m.gKronMagErr,
            m.gPSFMag,
            m.gPSFMagErr,
            a.gpsfLikelihood,
            m.ginfoFlag2,
            m.primaryDetection,
            ROW_NUMBER() OVER (PARTITION BY o.objID ORDER BY m.primaryDetection DESC) AS rn
        FROM ObjectThin o
        INNER JOIN StackObjectThin m ON o.objID = m.objID
        INNER JOIN StackObjectAttributes a ON o.objID = a.objID
        WHERE (m.ginfoFlag2 & 4) = 0
        AND o.raMean BETWEEN 96.21655788 AND 96.22275788
            AND o.decMean BETWEEN 13.31354339 AND 13.31974339
        AND (o.nStackDetections > 0 OR o.nDetections > 1)
    ),
    r_band AS (
        SELECT
            o.objID,
            o.raMean,
            o.decMean,
            m.rKronMag,
            m.rKronMagErr,
            m.rPSFMag,
            m.rPSFMagErr,
            a.rpsfLikelihood,
            m.rinfoFlag2,
            m.primaryDetection,
            ROW_NUMBER() OVER (PARTITION BY o.objID ORDER BY m.primaryDetection DESC) AS rn
        FROM ObjectThin o
        INNER JOIN StackObjectThin m ON o.objID = m.objID
        INNER JOIN StackObjectAttributes a ON o.objID = a.objID
        WHERE (m.rinfoFlag2 & 4) = 0
        AND o.raMean BETWEEN 96.21655788 AND 96.22275788
            AND o.decMean BETWEEN 13.31354339 AND 13.31974339
        AND (o.nStackDetections > 0 OR o.nDetections > 1)
    ),
    i_band AS (
        SELECT
            o.objID,
            o.raMean,
            o.decMean,
            m.iKronMag,
            m.iKronMagErr,
            m.iPSFMag,
            m.iPSFMagErr,
            a.ipsfLikelihood,
            m.iinfoFlag2,
            m.primaryDetection,
            ROW_NUMBER() OVER (PARTITION BY o.objID ORDER BY m.primaryDetection DESC) AS rn
        FROM ObjectThin o
        INNER JOIN StackObjectThin m ON o.objID = m.objID
        INNER JOIN StackObjectAttributes a ON o.objID = a.objID
        WHERE (m.iinfoFlag2 & 4) = 0
        AND o.raMean BETWEEN 96.21655788 AND 96.22275788
            AND o.decMean BETWEEN 13.31354339 AND 13.31974339
        AND (o.nStackDetections > 0 OR o.nDetections > 1)
    ),

g_table AS (
        SELECT * FROM g_band WHERE rn = 1
    ),
    r_table AS (
        SELECT * FROM r_band WHERE rn = 1
    ),
    i_table AS (
        SELECT * FROM i_band WHERE rn = 1
    ),
   

	obj_list AS (
    SELECT g_table.objID
        UNION
    SELECT r_table.objID
        UNION
    SELECT i_table.objID
        )

    SELECT
        obj_list.objID,
        COALESCE(g_table.raMean, r_table.raMean, i_table.raMean) AS raMean,
        COALESCE(g_table.decMean, r_table.decMean, i_table.decMean) AS decMean,
        g_table.gKronMag,
        g_table.gKronMagErr,
        g_table.gPSFMag,
        g_table.gPSFMagErr,
        g_table.gpsfLikelihood,
        g_table.ginfoFlag2,
        r_table.rKronMag,
        r_table.rKronMagErr,
        r_table.rPSFMag,
        r_table.rPSFMagErr,
        r_table.rpsfLikelihood,
        r_table.rinfoFlag2,
        i_table.iKronMag,
        i_table.iKronMagErr,
        i_table.iPSFMag,
        i_table.iPSFMagErr,
        i_table.ipsfLikelihood,
        i_table.iinfoFlag2
        INTO mydb.pstarr_sources_ra96p219_96p219_dec13p316_13p316
        FROM obj_list
        LEFT JOIN g_table ON obj_list.objID = g_table.objID
LEFT JOIN r_table ON obj_list.objID = r_table.objID
LEFT JOIN i_table ON obj_list.objID = i_table.objID;
"""

query = """
WITH
    g_band AS (
        SELECT
            o.objID,
            o.raMean,
            o.decMean,
            m.gKronMag,
            m.gKronMagErr,
            m.gPSFMag,
            m.gPSFMagErr,
            a.gpsfLikelihood,
            m.ginfoFlag2,
            m.primaryDetection,
            ROW_NUMBER() OVER (PARTITION BY o.objID ORDER BY m.primaryDetection DESC) AS rn
        FROM ObjectThin o
        INNER JOIN StackObjectThin m ON o.objID = m.objID
        INNER JOIN StackObjectAttributes a ON o.objID = a.objID
        WHERE (m.ginfoFlag2 & 4) = 0
        AND o.raMean BETWEEN 96.21655788 AND 96.22275788
            AND o.decMean BETWEEN 13.31354339 AND 13.31974339
        AND (o.nStackDetections > 0 OR o.nDetections > 1)
    ),
    r_band AS (
        SELECT
            o.objID,
            o.raMean,
            o.decMean,
            m.rKronMag,
            m.rKronMagErr,
            m.rPSFMag,
            m.rPSFMagErr,
            a.rpsfLikelihood,
            m.rinfoFlag2,
            m.primaryDetection,
            ROW_NUMBER() OVER (PARTITION BY o.objID ORDER BY m.primaryDetection DESC) AS rn
        FROM ObjectThin o
        INNER JOIN StackObjectThin m ON o.objID = m.objID
        INNER JOIN StackObjectAttributes a ON o.objID = a.objID
        WHERE (m.rinfoFlag2 & 4) = 0
        AND o.raMean BETWEEN 96.21655788 AND 96.22275788
            AND o.decMean BETWEEN 13.31354339 AND 13.31974339
        AND (o.nStackDetections > 0 OR o.nDetections > 1)
    ),
    i_band AS (
        SELECT
            o.objID,
            o.raMean,
            o.decMean,
            m.iKronMag,
            m.iKronMagErr,
            m.iPSFMag,
            m.iPSFMagErr,
            a.ipsfLikelihood,
            m.iinfoFlag2,
            m.primaryDetection,
            ROW_NUMBER() OVER (PARTITION BY o.objID ORDER BY m.primaryDetection DESC) AS rn
        FROM ObjectThin o
        INNER JOIN StackObjectThin m ON o.objID = m.objID
        INNER JOIN StackObjectAttributes a ON o.objID = a.objID
        WHERE (m.iinfoFlag2 & 4) = 0
        AND o.raMean BETWEEN 96.21655788 AND 96.22275788
            AND o.decMean BETWEEN 13.31354339 AND 13.31974339
        AND (o.nStackDetections > 0 OR o.nDetections > 1)
    ),
    g_table AS (
        SELECT * FROM g_band WHERE rn = 1
    ),
    r_table AS (
        SELECT * FROM r_band WHERE rn = 1
    ),
    i_table AS (
        SELECT * FROM i_band WHERE rn = 1
    ),

    obj_list AS (
        SELECT objID FROM g_table
        UNION
        SELECT objID FROM r_table
        UNION
        SELECT objID FROM i_table
    )


    SELECT
        obj_list.objID,
        COALESCE(g_table.raMean, r_table.raMean, i_table.raMean) AS raMean,
        COALESCE(g_table.decMean, r_table.decMean, i_table.decMean) AS decMean,
        g_table.gKronMag,
        g_table.gKronMagErr,
        g_table.gPSFMag,
        g_table.gPSFMagErr,
        g_table.gpsfLikelihood,
        g_table.ginfoFlag2,
        r_table.rKronMag,
        r_table.rKronMagErr,
        r_table.rPSFMag,
        r_table.rPSFMagErr,
        r_table.rpsfLikelihood,
        r_table.rinfoFlag2,
        i_table.iKronMag,
        i_table.iKronMagErr,
        i_table.iPSFMag,
        i_table.iPSFMagErr,
        i_table.ipsfLikelihood,
        i_table.iinfoFlag2

        INTO mydb.pstarr_sources_ra96p219_96p219_dec13p316_13p316

FROM obj_list
LEFT JOIN g_table ON obj_list.objID = g_table.objID
LEFT JOIN r_table ON obj_list.objID = r_table.objID
LEFT JOIN i_table ON obj_list.objID = i_table.objID;
"""
query = """WITH
    base AS (
        SELECT DISTINCT o.objID, o.raMean, o.decMean
        FROM ObjectThin o
        WHERE o.raMean BETWEEN 96.21655788 AND 96.22275788
            AND o.decMean BETWEEN 13.31354339 AND 13.31974339
            AND (o.nStackDetections > 0 OR o.nDetections > 1)
    ),
    g_band AS (
        SELECT
            o.objID,
            m.gKronMag,
            m.gKronMagErr,
            m.gPSFMag,
            m.gPSFMagErr,
            a.gpsfLikelihood,
            m.ginfoFlag2,
            ROW_NUMBER() OVER (
                PARTITION BY o.objID 
                ORDER BY 
                    CASE WHEN m.ginfoFlag2 = 0 THEN 0 ELSE 1 END,
                    m.primaryDetection DESC
            ) AS rn
        FROM base o
        INNER JOIN StackObjectThin m ON o.objID = m.objID
        INNER JOIN StackObjectAttributes a ON o.objID = a.objID
        WHERE (m.ginfoFlag2 & 4) = 0
    ),
    r_band AS (
        SELECT
            o.objID,
            m.rKronMag,
            m.rKronMagErr,
            m.rPSFMag,
            m.rPSFMagErr,
            a.rpsfLikelihood,
            m.rinfoFlag2,
            ROW_NUMBER() OVER (
                PARTITION BY o.objID 
                ORDER BY 
                    CASE WHEN m.rinfoFlag2 = 0 THEN 0 ELSE 1 END,
                    m.primaryDetection DESC
            ) AS rn
        FROM base o
        INNER JOIN StackObjectThin m ON o.objID = m.objID
        INNER JOIN StackObjectAttributes a ON o.objID = a.objID
        WHERE (m.rinfoFlag2 & 4) = 0
    ),
    i_band AS (
        SELECT
            o.objID,
            m.iKronMag,
            m.iKronMagErr,
            m.iPSFMag,
            m.iPSFMagErr,
            a.ipsfLikelihood,
            m.iinfoFlag2,
            ROW_NUMBER() OVER (
                PARTITION BY o.objID 
                ORDER BY 
                    CASE WHEN m.iinfoFlag2 = 0 THEN 0 ELSE 1 END,
                    m.primaryDetection DESC
            ) AS rn
        FROM base o
        INNER JOIN StackObjectThin m ON o.objID = m.objID
        INNER JOIN StackObjectAttributes a ON o.objID = a.objID
        WHERE (m.iinfoFlag2 & 4) = 0
    ),
    g_table AS (
        SELECT * FROM g_band WHERE rn = 1
    ),
    r_table AS (
        SELECT * FROM r_band WHERE rn = 1
    ),
    i_table AS (
        SELECT * FROM i_band WHERE rn = 1
    )

SELECT
    base.objID,
    base.raMean,
    base.decMean,
    g_table.gKronMag,
    g_table.gKronMagErr,
    g_table.gPSFMag,
    g_table.gPSFMagErr,
    g_table.gpsfLikelihood,
    g_table.ginfoFlag2,
    r_table.rKronMag,
    r_table.rKronMagErr,
    r_table.rPSFMag,
    r_table.rPSFMagErr,
    r_table.rpsfLikelihood,
    r_table.rinfoFlag2,
    i_table.iKronMag,
    i_table.iKronMagErr,
    i_table.iPSFMag,
    i_table.iPSFMagErr,
    i_table.ipsfLikelihood,
    i_table.iinfoFlag2

INTO mydb.testing

FROM base
LEFT JOIN g_table ON base.objID = g_table.objID
LEFT JOIN r_table ON base.objID = r_table.objID
LEFT JOIN i_table ON base.objID = i_table.objID;
"""

jobid = jobs.submit(query, task_name="test", context='PanSTARRS_DR2')
print('submitted!')
jobs.monitor(jobid)
print('done!')

# from casjobs import CasJobs

# test_query = "SELECT TOP 10 * FROM ObjectThin"
# wsid, password = get_credentials('mast_login.txt')
# jobs = CasJobs(userid=wsid, password=password, request_type='POST')
# job_id = jobs.submit(test_query, context='PanSTARRS_DR2')
# status = jobs.monitor(job_id)
